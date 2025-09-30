# utils/offline_plots.py
import json
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pytorch_lightning import Callback, Trainer


class OfflinePlotsCallback(Callback):
    """
    Collect metrics when --no_wandb is passed and dump them in
    <run_dir>/offline_report_<timestamp>.pdf and a JSON with config + results.

    JSON structure produced:

    {
      "config": { ... },
      "history": {
        "train_loss": [...],     # baseline (index 0) + per epoch
        "train_acc":  [...],     # baseline (index 0) + per epoch
        "val_loss":   [...],     # baseline (index 0) + per epoch
        "val_acc":    [...],     # baseline (index 0) + per epoch
        "comm_cost":  [...]      # per epoch (cumulative MB after each epoch)
      },
      "val_acc":  <final scalar>,
      "test_acc": <final scalar>,
      "val_loss": <final scalar>,
      "test_loss":<final scalar>,
      "comm_cost":<final scalar>,   # final cumulative MB
      "grad_squared_norm": <scalar or null>,
      "has_baseline": true
    }
    """

    def __init__(self, run_dir: str, num_clients: int, config_dict=None, meta: dict | None = None):
        super().__init__()
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.num_clients = int(num_clients)
        self.config_dict = config_dict or {}
        self.meta = meta or {}

        # Per-epoch histories (lists)
        self.train_loss = []
        self.train_acc  = []
        self.val_loss   = []   # includes baseline at index 0
        self.val_acc    = []   # includes baseline at index 0
        self.comm_cost  = []   # cumulative MB after each train epoch

        # For the connectivity figure (1 entry per epoch)
        self.connectivity_log = []

        # Optional extras
        self.grad_squared_norm = []

        # Final values to surface at top level
        self.final_val_loss  = None
        self.final_val_acc   = None
        self.final_test_loss = None
        self.final_test_acc  = None
        self.final_comm_cost = None

        # Plotting control
        self.include_baseline = True

    # -------------------------------
    # Helpers
    # -------------------------------
    @torch.no_grad()
    def _compute_over_loader(self, pl_module, loader):
        """Sample-weighted CE loss and accuracy over a loader (no optimizer touches)."""
        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval()

        total_loss = 0.0
        total_correct = 0
        total = 0

        for batch in loader:
            # Allow (x,y) or (x,y,*) batches
            if len(batch) < 2:
                continue
            x, y = batch[0].to(device), batch[1].to(device)
            logits = pl_module(x)
            total_loss += F.cross_entropy(logits, y, reduction="sum").item()
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()

        if was_training:
            pl_module.train()
        if total == 0:
            return None, None
        return total_loss / total, total_correct / total

    # -------------------------------
    # PL hooks
    # -------------------------------
    def on_fit_start(self, trainer: Trainer, pl_module):
        """Compute *baseline* train & val before any training (epoch 0, chance-ish)."""
        dm = trainer.datamodule

        # Baseline TRAIN (deterministic loader over the full train set)
        if hasattr(dm, "eval_train_dataloader"):
            eval_train_loader = dm.eval_train_dataloader()
        else:
            bs = getattr(dm, "batch_size", None) or 128
            nw = getattr(dm, "num_workers", 0)
            eval_train_loader = DataLoader(
                dm.train_dataset, batch_size=bs, num_workers=nw,
                shuffle=False, drop_last=False
            )
        tl, ta = self._compute_over_loader(pl_module, eval_train_loader)
        if tl is not None:
            self.train_loss.append(float(tl))
            self.train_acc.append(float(ta))

        # Baseline VAL
        if dm and hasattr(dm, "val_dataloader"):
            vloader = dm.val_dataloader()
            if vloader:
                vl, va = self._compute_over_loader(pl_module, vloader)
                if vl is not None:
                    self.val_loss.append(float(vl))
                    self.val_acc.append(float(va))


    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.state.fn != "fit" or getattr(trainer, "sanity_checking", False):
            return
        dm = trainer.datamodule

        # VAL
        vloader = dm.val_dataloader()
        vl, va = self._compute_over_loader(pl_module, vloader)
        if vl is not None:
            self.val_loss.append(float(vl))
            self.val_acc.append(float(va))

        # TRAIN (full 60k, same pipeline)
        tloader_full = (dm.eval_train_dataloader()
                        if hasattr(dm, "eval_train_dataloader") else dm.train_dataloader())
        tl, ta = self._compute_over_loader(pl_module, tloader_full)
        if tl is not None:
            self.train_loss.append(float(tl))
            self.train_acc.append(float(ta))

        # TRAIN (size-matched subset = |val|)
        if hasattr(dm, "eval_train_subset_dataloader"):
            tloader_5k = dm.eval_train_subset_dataloader(size=len(dm.val_dataset),
                                                        seed=getattr(dm, "seed", 0))
            tl5, ta5 = self._compute_over_loader(pl_module, tloader_5k)
            if tl5 is not None:
                if not hasattr(self, "train_loss_5k"):
                    self.train_loss_5k, self.train_acc_5k = [], []
                self.train_loss_5k.append(float(tl5))
                self.train_acc_5k.append(float(ta5))

        
    def on_train_epoch_end(self, trainer: Trainer, pl_module):
        """Record which clients were online this epoch (for the connectivity plot)."""
        cm = trainer.callback_metrics
        if "comm_cost"  in cm: self.comm_cost .append(cm["comm_cost"].item())
        if "grad_squared_norm" in cm:
            self.grad_squared_norm.append(cm["grad_squared_norm"].item())
        seen = getattr(pl_module, "online_seen_this_epoch", None)
        if seen is not None:
            self.connectivity_log.append(sorted(list(seen)))
        else:
            # Fallback: snapshot of whoever is currently connected
            try:
                self.connectivity_log.append(list(pl_module._connected_now()))
            except Exception:
                self.connectivity_log.append([])

    def on_test_end(self, trainer: Trainer, pl_module):
        """Compute final test metrics and write the PDF + JSON."""
        dm = trainer.datamodule
        if dm and hasattr(dm, "test_dataloader"):
            try:
                dm.setup("test")
            except Exception:
                pass
            tloader = dm.test_dataloader()
            if tloader:
                tl, ta = self._compute_over_loader(pl_module, tloader)
                if tl is not None:
                    self.final_test_loss = float(tl)
                    self.final_test_acc  = float(ta)

        # Final val metrics = last entries we saw during training
        if self.val_loss: self.final_val_loss = float(self.val_loss[-1])
        if self.val_acc:  self.final_val_acc  = float(self.val_acc[-1])

        # Final comm cost: last cumulative value we saw
        if self.comm_cost:
            self.final_comm_cost = float(self.comm_cost[-1])
        else:
            # fallback to whatever is logged at test time (rare)
            if "comm_cost" in trainer.callback_metrics:
                self.final_comm_cost = float(trainer.callback_metrics["comm_cost"].item())
            else:
                self.final_comm_cost = 0.0

        self._write_pdf()
        self._dump_results()

    # -------------------------------
    # Reporting
    # -------------------------------
    def _write_pdf(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pdf_path = self.run_dir / f"offline_report_{stamp}.pdf"

        # how many *epochs* (not counting baseline) to show
        n_plot = int(
            self.meta.get("max_epochs")
            or self.config_dict.get("trainer", {}).get("max_epochs", len(self.train_loss))
        )

        # Build plot series (drop the baseline for plotting if requested)
        train_loss_plot = self.train_loss[1:1 + n_plot] if self.include_baseline else self.train_loss[:n_plot]
        train_acc_plot  = self.train_acc [1:1 + n_plot] if self.include_baseline else self.train_acc [:n_plot]
        val_loss_plot   = self.val_loss  [1:1 + n_plot] if self.include_baseline else self.val_loss  [:n_plot]
        val_acc_plot    = self.val_acc   [1:1 + n_plot] if self.include_baseline else self.val_acc   [:n_plot]

        # x axes
        x_train = np.arange(len(train_loss_plot))
        x_val   = np.arange(len(val_loss_plot))

        # connectivity (one row per training epoch)
        n_epochs = int(len(self.train_loss) - (1 if self.include_baseline else 0))
        conn_mat = np.zeros((max(n_epochs, 0), int(self.num_clients)), dtype=np.int8)
        for ep, online in enumerate(self.connectivity_log[:n_epochs]):
            if len(online):
                conn_mat[ep, np.asarray(online, dtype=int)] = 1
        online_per_epoch = conn_mat.sum(1) if conn_mat.size else np.array([])

        # timeline data
        intervals_per_client = [[] for _ in range(int(self.num_clients))]
        for ep, online in enumerate(self.connectivity_log[:n_epochs]):
            for cid in online:
                intervals_per_client[int(cid)].append((ep, 1))

        with PdfPages(pdf_path) as pdf:
            # Header page
            header_lines = [
                f"batching       : {self.meta.get('batching','?')}",
                f"participation  : {self.meta.get('participation','?')}",
                f"cutsize        : {self.meta.get('cutsize','?')}",
                f"scheduler      : {self.meta.get('scheduler','?')}",
                f"compressor     : {self.meta.get('compressor','none')}",
            ]
            if self.meta.get("compression_p") is not None:
                header_lines[-1] += f" (p={self.meta['compression_p']})"
            details = "\n".join(header_lines)

            fig = plt.figure(figsize=(8, 3.0))
            title = self.config_dict.get("logging", {}).get("experiment_name", "Offline Report")
            txt_final = []
            if self.final_test_acc is not None:
                txt_final.append(f"Final Test Acc = {self.final_test_acc:.3f}")
            if self.final_comm_cost is not None:
                txt_final.append(f"Comm Cost = {self.final_comm_cost:.3f} MB")
            fig.text(0.5, 0.78, title, ha="center", va="center", fontsize=16, weight="bold")
            fig.text(0.5, 0.52, " | ".join(txt_final), ha="center", va="center", fontsize=12)
            fig.text(0.01, 0.05, details, ha="left", va="bottom", fontsize=9, family="monospace")
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

            # Loss curves
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(x_train, train_loss_plot, label="train")
            ax.plot(x_val,   val_loss_plot,   label="val")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_yscale("log")
            ax.grid(True); ax.legend()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            pdf.savefig(fig); plt.close(fig)

            # Accuracy curves
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(x_train, train_acc_plot, label="train")
            ax.plot(x_val,   val_acc_plot,   label="val")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.grid(True); ax.legend()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            pdf.savefig(fig); plt.close(fig)

            # Connectivity timeline + counts
            fig, (ax_tl, ax_cnt) = plt.subplots(
                2, 1, figsize=(15, 5.3),
                gridspec_kw={"height_ratios": [3, 1]}, sharex=True
            )
            for cid, ivals in enumerate(intervals_per_client):
                if ivals:
                    ax_tl.broken_barh(ivals, (cid - .4, .8),
                                      facecolors="forestgreen", edgecolors="none")
            ax_tl.set_ylim(-.5, int(self.num_clients) - .5)
            ax_tl.set_xlim(0, max(n_epochs, 1))
            ax_tl.set_yticks(range(int(self.num_clients)))
            ax_tl.set_yticklabels([f"Client {c}" for c in range(int(self.num_clients))])
            ax_tl.grid(axis="x", linestyle=":", color="grey", alpha=.4)
            ax_tl.set_title("Connectivity timeline  (green = online)")
            ax_tl.legend(handles=[plt.Line2D([], [], marker="s", color="forestgreen",
                                             linestyle="", markersize=9, label="online")],
                         loc="upper right", frameon=False)

            ax_cnt.bar(np.arange(len(online_per_epoch)), online_per_epoch, width=1, edgecolor="grey")
            ax_cnt.set_ylabel("# online")
            ax_cnt.set_ylim(0, int(self.num_clients))
            ax_cnt.set_xlabel("Epoch")
            ax_cnt.set_xticks(np.arange(0, len(online_per_epoch) + 1, 10))
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig); plt.close(fig)

        print(f"\n📝  offline_report written to {pdf_path.resolve()}\n")

    def _dump_results(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = self.run_dir / f"summary_{timestamp}.json"

        history = {
            "train_loss": self.train_loss,  # includes baseline at index 0
            "train_acc":  self.train_acc,   # includes baseline at index 0
            "val_loss":   self.val_loss,    # includes baseline at index 0
            "val_acc":    self.val_acc,     # includes baseline at index 0
            "comm_cost":  self.comm_cost,   # cumulative MB per epoch (after epoch end)
        }
        if hasattr(self, "train_loss_5k"):
            history["train_loss_5k"] = self.train_loss_5k
            history["train_acc_5k"]  = self.train_acc_5k

        summary = {
            "config": self.config_dict,
            "history": history,

            # Final scalar metrics at the top-level
            "val_acc":  (None if self.final_val_acc  is None else float(self.final_val_acc)),
            "test_acc": (None if self.final_test_acc is None else float(self.final_test_acc)),
            "val_loss": (None if self.final_val_loss is None else float(self.final_val_loss)),
            "test_loss":(None if self.final_test_loss is None else float(self.final_test_loss)),
            "comm_cost":(None if self.final_comm_cost is None else float(self.final_comm_cost)),

            "grad_squared_norm": (self.grad_squared_norm[-1] if self.grad_squared_norm else None),
            "has_baseline": True,
        }

        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Saved JSON summary to {out.resolve()}")
