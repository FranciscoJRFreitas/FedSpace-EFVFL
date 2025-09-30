import torch
import torch.nn.functional as F
import pytorch_lightning as L
from torchmetrics import Accuracy
import numpy as np
from pathlib import Path
import torch.nn.utils as tgutils
import datetime

torch.set_float32_matmul_precision("high")

optimizers_d = {"sgd": torch.optim.SGD}
schedulers_d = {"cosine_annealing_lr": torch.optim.lr_scheduler.CosineAnnealingLR}

class SplitNN(L.LightningModule):
    def __init__(self, representation_models, fusion_model, lr, momentum, weight_decay,
                optimizer, eta_min_ratio, scheduler, num_epochs,
                private_labels, batch_size, compute_grad_sqd_norm, connectivity_schedule=None,
                fedspace_scheduler=None, partial_participation: bool = True, agg_mode: str = "sync", buffer_M: int = 96):
        super().__init__()
        self.slot_counter = 0
        self.save_hyperparameters(ignore=["representation_models", "fusion_model"])

        self.automatic_optimization = False
        self.initial_grad_norm = None
        self.num_clients = len(representation_models)
        self.n_mbytes = 0

        self.representation_models = representation_models
        self.fusion_model = fusion_model

        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.fusion_model.num_classes)
        self.val_accuracy   = Accuracy(task="multiclass", num_classes=self.fusion_model.num_classes)
        self.test_accuracy  = Accuracy(task="multiclass", num_classes=self.fusion_model.num_classes)
        self.compute_grad_sqd_norm = compute_grad_sqd_norm

        self.connectivity_schedule = connectivity_schedule or {}
        self._num_slots = len(self.connectivity_schedule) or 1
        print(f"[SplitNN] {self._num_slots} connectivity slots")
        self.prev_repr = [None] * self.num_clients
        self.fedspace_scheduler = fedspace_scheduler

        self._agg_mask_epoch = []          # 96-length per epoch
        self._agg_masks = []               # list of epochs
        self.bytes_per_round = []          # MB per aggregation (append on aggregate)

        if self.fedspace_scheduler is not None:
            assert getattr(self.fedspace_scheduler, "K", self.num_clients) == self.num_clients, \
                f"K mismatch: scheduler.K={getattr(self.fedspace_scheduler,'K',None)} vs num_clients={self.num_clients}"

        self.agg_round = 0
        self.staleness_vec = []
        self.last_upload_round = [-1] * self.num_clients
        self.staleness_this_round = None
        self.loss_per_round = []
        self._window_T = 1.0

        self._mb_in_slot = 0                      # batches elapsed in current slot
        self._slot_do_update = True               # slot-level decision
        self.global_batch_idx = 0
        self.partial_participation = partial_participation
        self._upload_buffer = set()

        self._ci_this_slot = []                   # clients online (C_i) in current slot
        self.online_seen_this_epoch = set()       # for logging

        self._val_loss_sum = 0.0
        self._val_count    = 0
        self._test_loss_sum = 0.0
        self._test_count    = 0

        self.agg_mode   = str(agg_mode).lower()   # "sync" | "async" | "fedbuff" | "fedspace"
        self.buffer_M   = int(buffer_M)

    def _connected_now(self):
        """Return list of clients whose link is up in *this* slot."""
        if not self.partial_participation:
            return list(range(self.num_clients))
        slot = self.slot_counter % self._num_slots
        return self.connectivity_schedule.get(slot, list(range(self.num_clients)))

    def on_train_start(self):
        S = max(1, self._num_slots)
        nb = len(self.trainer.datamodule.train_dataloader())
        q, r = divmod(nb, S)
        self._slot_sizes = [q + (1 if s < r else 0) for s in range(S)]
        self.batches_per_slot = self._slot_sizes[0]
        self.print(f"[SplitNN] batches_per_slot={self.batches_per_slot}  (epoch_batches={nb}, slots={S})")
 
    def on_train_epoch_start(self):
        self.online_seen_this_epoch.clear()
        self._agg_mask_epoch = []

    def on_validation_epoch_start(self):
        self._val_loss_sum = 0.0
        self._val_count    = 0
        self.val_accuracy.reset()

    def forward(self, x):
        representations = [model(self.get_feature_block(x, i)) for i, model in enumerate(self.representation_models)]
        return self.fusion_model(representations)

    def training_step(self, batch, batch_idx=None):
        try:
            x, y, indices = batch
            optimizers = self.optimizers()

            if self._mb_in_slot == 0:
                if self.agg_mode == "fedspace" and self.fedspace_scheduler and self.fedspace_scheduler.is_window_start():
                    self._window_T = float(self.trainer.callback_metrics.get("val_loss", 1.0))
                    self.fedspace_scheduler.begin_next_plan(self._window_T)
                    self.fedspace_scheduler.plan_next_step(budget=32)
                    self._debug_planned_a = (self.fedspace_scheduler.current_a.copy()
                                            if getattr(self.fedspace_scheduler, "current_a", None) is not None else None)

                C_i = self._connected_now()
                self._ci_this_slot = C_i
                self.online_seen_this_epoch.update(C_i)

                # --- Buffer update policy ---
                if self.agg_mode == "async":
                    # async: buffer = current arrivals only (no union)
                    self._upload_buffer = set(C_i)
                else:
                    # sync, fedbuff, fedspace: union since last aggregate
                    self._upload_buffer.update(C_i)

                if self.agg_mode == "fedspace":
                    loss_T = float(self._window_T)
                    self._slot_do_update = (
                        self.fedspace_scheduler is None
                        or self.fedspace_scheduler.should_aggregate(C_i, loss_T)
                    )
                elif self.agg_mode == "sync":
                    # wait for ALL K clients to be in the buffer
                    self._slot_do_update = (len(self._upload_buffer) == self.num_clients)
                elif self.agg_mode == "fedbuff":
                    # aggregate when buffer reaches M unique clients
                    self._slot_do_update = (len(self._upload_buffer) >= self.buffer_M)
                else:  # "async"
                    # aggregate whenever we have any arrivals this slot
                    self._slot_do_update = (len(C_i) > 0)

                if self._slot_do_update:
                    self.zero_grad(set_to_none=True)

                self.print(f"--- slot {self.slot_counter:02d} C_i={C_i} |R|={len(self._upload_buffer)} "
                        f"⇒ {'AGGREGATE' if self._slot_do_update else 'SKIP'}")


            C_i = self._ci_this_slot

            compressed_representations = []
            do_compress = bool(getattr(self.representation_models[0].compression_module, "compressor", None))

            is_mean = getattr(self.fusion_model, "aggregation_mechanism", "") == "mean"
            is_sum  = getattr(self.fusion_model, "aggregation_mechanism", "") == "sum"

            with torch.no_grad():
                for i, model in enumerate(self.representation_models):
                    if i in C_i:  # online → fresh
                        rep = model(self.get_feature_block(x, i),
                                    apply_compression=do_compress,
                                    indices=indices,
                                    epoch=self.slot_counter)
                        self.prev_repr[i] = rep.detach()
                    else:
                        rep = torch.zeros(
                            x.size(0),
                            model.compression_module.cut_size,
                            device=x.device,
                            dtype=x.dtype,
                        )
                    compressed_representations.append(rep.detach())

            is_last_batch_of_slot = (self._mb_in_slot == self.batches_per_slot - 1)
            if (self.agg_mode == "fedspace" and self.fedspace_scheduler is not None
        and is_last_batch_of_slot and getattr(self.fedspace_scheduler, "_plan_state", None) is not None):
                self.fedspace_scheduler.plan_next_step(budget=32)

            S_t = set(C_i) if self.agg_mode == "async" else set(self._upload_buffer)

            alpha = float(getattr(self.hparams, "staleness_alpha", 0.0))
            lam = {}
            if alpha > 0.0 and S_t:
                raw = []
                idxs = sorted(S_t)
                for k in idxs:
                    last = self.last_upload_round[k]
                    s_k = 0 if last < 0 else (self.agg_round - last - 1)
                    raw.append((s_k + 1.0) ** (-alpha))
                Z = float(sum(raw)) or float(len(idxs))
                lam = {k: float(r / Z) for k, r in zip(idxs, raw)}

            if self._slot_do_update:
                # scale so one end-of-slot step is comparable
                if is_mean:
                    passes_total = 1
                else:
                    passes_total = max(1, len(S_t))
                scale = 1.0 / max(1, self.batches_per_slot * passes_total)

                # online client pass to update that client + fusion
                for k in sorted(S_t):
                    if k in self._ci_this_slot:
                        rep_k = self.representation_models[k](
                            self.get_feature_block(x, k), apply_compression=False
                        )
                    else:
                        if self.prev_repr[k] is None:
                            continue
                        rep_k = self.prev_repr[k].detach()
                    mixed = [rep.detach() for rep in compressed_representations]
                    mixed[k] = rep_k
                    out_k = self.fusion_model(mixed)
                    w_k = float(lam.get(k, 1.0 / max(1, len(S_t)))) if lam else (1.0 / max(1, len(S_t)))
                    loss_k = w_k * F.cross_entropy(out_k, y) * scale
                    self.manual_backward(loss_k)

                if not lam:
                    # one fusion-only pass with all (compressed) reps
                    out_all = self.fusion_model([rep.detach() for rep in compressed_representations])
                    # lam sums to 1 over S_t, so scalar weight can be 1.0
                    w_all = 1.0 if lam else 1.0
                    loss_all = w_all * F.cross_entropy(out_all, y) * scale
                    self.manual_backward(loss_all)

            # ── if NOT last batch of slot, just keep accumulating
            if not is_last_batch_of_slot:
                return

            self._agg_mask_epoch.append(1 if self._slot_do_update else 0)

            # ── end of slot: if SKIP → nothing else to do
            if not self._slot_do_update:
                # do not clear upload buffer; it accumulates across skipped slots
                return

            # aggregation happens here

            if self.agg_mode == "async":
                # async: aggregate only with this slot's arrivals
                uploaded = set(self._ci_this_slot)
            else:
                # sync, fedbuff, fedspace: union since last aggregate
                uploaded = set(self._upload_buffer)


            # (1) COMM: count bytes ONLY on aggregation, for the union buffer
            if uploaded:
                bits = 0
                for k in uploaded:
                    rep = self.prev_repr[k]
                    if rep is not None:
                        bits += self._calculate_n_bits([rep])
                mb = bits / 8e6
                self.n_mbytes += mb
                self.bytes_per_round.append(mb)

            if getattr(self.representation_models[0].compression_module, "compressor", None) is None:
                B = self.prev_repr[next(iter(uploaded))].shape[0] if uploaded else 0
                expected = len(uploaded) * B * self.representation_models[0].compression_module.cut_size * 4 / 1e6
                if abs(mb - expected) / max(1e-9, expected) > 0.05:   # >5%
                    self.print(f"[WARN] bytes/round mismatch: mb={mb:.3f} expected={expected:.3f} "
                            f"(uploaded={len(uploaded)}, B={B})")

            # (2) record upload info
            if self.agg_mode == "fedspace" and self.fedspace_scheduler is not None and uploaded:
                self.fedspace_scheduler.record_upload(uploaded)

            # (3) staleness + round index (credit only those in 'uploaded')
            staleness = []
            for k in range(self.num_clients):
                if k not in uploaded:
                    staleness.append(-1)
                else:
                    last = self.last_upload_round[k]
                    s = 0 if last < 0 else self.agg_round - last - 1
                    staleness.append(s)
                    self.last_upload_round[k] = self.agg_round
            self.staleness_this_round = staleness
            self.agg_round += 1
            self.staleness_vec.append(staleness)

            # (4) ONE optimizer step per slot
            tgutils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            # step only online clients (+ fusion)
            for i in C_i:
                optimizers[i].step()
            optimizers[-1].step()  # fusion

            # ready for next slot
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            this_slot_uploaded = sorted(C_i)
            credited_on_aggregate = sorted(uploaded)
            self.print(f"[Round {self.agg_round:3d}] slot={self.slot_counter:02d}  "
                    f"C_i={this_slot_uploaded}  uploaded(union)={credited_on_aggregate}")

            # eval loss per aggregation round - for offline regressor
            with torch.no_grad():
                out_eval = self.fusion_model([rep.detach() for rep in compressed_representations])
                self.loss_per_round.append(F.cross_entropy(out_eval, y).item())

            # clear buffer after successful aggregation
            if self.agg_mode in ("sync", "fedbuff", "fedspace"):
                self._upload_buffer.clear()

        finally:
            self._mb_in_slot += 1
            if self._mb_in_slot >= self.batches_per_slot:
                self._mb_in_slot = 0
                self.slot_counter = (self.slot_counter + 1) % self._num_slots
                if hasattr(self, "_slot_sizes"):
                    self.batches_per_slot = self._slot_sizes[self.slot_counter]

    def _calculate_n_bits(self, reps):
        n_bits = 0
        for rep in reps:
            bits_per_el = rep.element_size() * 8
            compressor = self.representation_models[0].compression_module.compressor
            if compressor is None:
                n_bits += rep.numel() * bits_per_el
            elif compressor == 'topk':
                n_bits += rep.numel() * bits_per_el * self.representation_models[0].compression_module.compression_parameter
            elif compressor == 'qsgd':
                n_bits += bits_per_el + rep.numel() * (1 + self.representation_models[0].compression_module.compression_parameter)
            else:
                raise ValueError(f"Unknown compressor: {compressor}")
        return n_bits

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.val_accuracy.update(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.log("val_acc",  self.val_accuracy, on_epoch=True, prog_bar=True, logger=True)

    def on_test_epoch_start(self):
        self._test_loss_sum = 0.0
        self._test_count    = 0
        self.test_accuracy.reset()

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss_sum = F.cross_entropy(y_hat, y, reduction="sum")
        self._test_loss_sum += loss_sum.item()
        self._test_count    += y.numel()
        self.test_accuracy.update(y_hat, y)
        return loss_sum
    
    def on_train_epoch_end(self):
        stepped = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.parameters())
        if stepped:
            for scheduler in self.lr_schedulers():
                scheduler.step()

        self.log('comm_cost', self.n_mbytes, on_epoch=True, prog_bar=True)

        if self._agg_mask_epoch:
            self._agg_masks.append(self._agg_mask_epoch)
            if hasattr(self, "_debug_planned_a") and self._debug_planned_a is not None:
                if len(self._debug_planned_a) == len(self._agg_mask_epoch):
                    diff = np.sum(np.abs(np.array(self._agg_mask_epoch) - self._debug_planned_a))
                    if diff:
                        self.print(f"[WARN] agg mask != planned pattern in this window (diff={diff})")
            self._agg_mask_epoch = []

    def on_train_end(self):
        if not self.staleness_vec:
            return
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = (Path(getattr(self.logger, "log_dir", self.trainer.default_root_dir)) / f"debug_{stamp}")
        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "staleness.npy",  np.array(self.staleness_vec,    dtype=np.int16))
        np.save(run_dir / "loss_round.npy", np.array(self.loss_per_round,    dtype=np.float32))
        np.save(run_dir / "bytes_round.npy",np.array(self.bytes_per_round,   dtype=np.float32))
        if self._agg_masks:
            np.save(run_dir / "agg_mask.npy", np.array(self._agg_masks, dtype=np.int8))
        self.print(f"Saved staleness, loss, bytes, mask to {run_dir}")

    def on_test_epoch_end(self):
        test_loss = self._test_loss_sum / max(1, self._test_count)
        test_acc  = self.test_accuracy.compute()
        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc",  test_acc,  on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        Optimizer = optimizers_d[self.hparams.optimizer]
        client_optimizers = [Optimizer(model.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
                            for model in self.representation_models]
        fusion_optimizer = Optimizer(self.fusion_model.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        optimizers = client_optimizers + [fusion_optimizer]

        if self.hparams.scheduler is None:
            return optimizers
        else:
            Scheduler = schedulers_d[self.hparams.scheduler]
            schedulers = [Scheduler(opt, T_max=self.hparams.num_epochs, eta_min=self.hparams.lr * self.hparams.eta_min_ratio) for opt in optimizers]
            return optimizers, schedulers

    def get_feature_block(self):
        raise NotImplementedError("Subclasses must implement this method")