#!/usr/bin/env python3
# -------------------------------------------------------------
import yaml, wandb
import pytorch_lightning as L
from pathlib import Path
from argparse import ArgumentParser, Namespace
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import requests

from utils.offline_plots  import OfflinePlotsCallback
from utils.utils          import load_module
from utils.connectivity   import load_connectivity
from utils.scheduler      import FedSpaceScheduler


def build_components(agg_mode: str,
                     schedule: dict[int, list[int]],
                     num_clients: int):
    """
    Return (scheduler_obj, connectivity_table, partial_flag).
    - fedspace: dynamic schedule (use regressor; unchanged)
    - sync/async/fedbuff: no scheduler, use provided connectivity
    - full: all clients online every slot (no partial participation)
    """
    if agg_mode == "fedspace":
        sched = FedSpaceScheduler(
            window_len = 96,
            regressor_pkl = "models/utility_regressor.pkl",
            connectivity_table = schedule,
            n_rand = 1024, n_min = 8, n_max = 32,
        )
        return sched, schedule, True

    if agg_mode == "full":
        full_table = {s: list(range(num_clients)) for s in schedule}
        return None, full_table, False

    return None, schedule, True

def collect_run_meta(dm, model, mode, batch_size, partial_flag, max_epochs, buffer_M):
    meta = {
        "batching"      : "minibatch" if batch_size < dm.num_train_samples else "full-batch",
        "participation" : "full" if not partial_flag else "partial",
        "scheduler"     : mode,  # fedspace | sync | async | fedbuff | full
        "compressor"    : getattr(model.representation_models[0].compression_module, "compressor", "none"),
        "compression_p" : getattr(model.representation_models[0].compression_module, "compression_parameter", None),
        "num_clients"   : model.num_clients,
        "batch_size"    : batch_size,
        "cutsize"       : getattr(model, "cut_size", None),
        "max_epochs"    : int(max_epochs),
        "buffer_M"      : int(buffer_M),
    }
    return meta

def run_one_seed(cfg: dict, seed: int, args: Namespace,
                 connectivity_schedule, num_clients):
    L.seed_everything(seed)

    wandb_logger = None
    if not args.no_wandb:
        wandb_logger = WandbLogger(
            project = cfg["logging"]["project_name"],
            name    = f"{cfg['logging']['experiment_name']}-s{seed}",
            save_dir = "./logs",
        )
        wandb_logger.experiment.config.update(cfg)

    dm_cls = load_module(cfg["data"]["module_path"],
                         cfg["data"]["module_name"])
    dm     = dm_cls(**cfg["data"]["params"])
    dm.prepare_data(); dm.setup("fit"); dm.setup("validate")

    mode = cfg["agg_mode"]  # "fedspace" | "sync" | "async" | "fedbuff" | "full"
    scheduler_obj, table, partial_flag = build_components(
        mode, connectivity_schedule, num_clients
    )

    model_cls = load_module(cfg["model"]["module_path"],
                            cfg["model"]["module_name"])
    model = model_cls(**cfg["model"]["params"],
                      num_samples = dm.num_train_samples,
                      batch_size  = dm.train_dataloader().batch_size,
                      num_epochs  = cfg["trainer"]["max_epochs"],
                      connectivity_schedule = table,
                      fedspace_scheduler    = scheduler_obj,
                      partial_participation = partial_flag,
                      agg_mode = cfg["agg_mode"],
                      buffer_M = cfg["buffer_M"])

    print(f"num_clients = {model.num_clients}",
          "local_input_size =", model.representation_models[0].fc.in_features)
    
    run_meta = collect_run_meta(
        dm, model, mode,
        dm.train_dataloader().batch_size,
        partial_flag,
        cfg["trainer"]["max_epochs"],
        cfg.get("buffer_M", 96),
    )

    run_dir = Path(f"results/{cfg['logging']['experiment_name']}-s{seed}")
    offline_cb = (OfflinePlotsCallback(run_dir, num_clients,cfg, run_meta) if args.no_wandb else None)

    trainer = Trainer(
        max_epochs = cfg["trainer"]["max_epochs"],
        logger     = wandb_logger or False,
        accelerator='gpu',
        devices    = args.gpu,
        log_every_n_steps = 1,
        enable_checkpointing = False,
        callbacks  = [offline_cb] if offline_cb else [],
        num_sanity_val_steps = 0,
    )

    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)

    if wandb_logger:
        wandb.finish()

def print_connectivity_stats(table, K):
    slots = sorted(table.keys())
    assert slots == list(range(len(slots))), f"slot keys must be 0..{len(slots)-1}"
    online_counts = [len(table[s]) for s in slots]
    seen = set()
    per_client = [0]*K
    for s in slots:
        for k in table[s]:
            seen.add(k); per_client[k]+=1
    print(f"[Conn] slots={len(slots)}  avg|C_i|={np.mean(online_counts):.1f}  "
          f"min/max|C_i|={min(online_counts)}/{max(online_counts)}  coverage={len(seen)}/{K}")
    worst = sorted(range(K), key=lambda k: per_client[k])[:5]
    best  = sorted(range(K), key=lambda k: per_client[k])[-5:]
    print("[Conn] least-online:", [(k,per_client[k]) for k in worst])
    print("[Conn] most-online :", [(k,per_client[k]) for k in best])

def main(args: Namespace):
    cfg_path = Path("configs") / args.config
    cfg = yaml.safe_load(cfg_path.read_text())

    connectivity_schedule, id2idx = load_connectivity(cfg["connectivity"]["path"])
    num_clients = len(id2idx)
    print(f"Using {num_clients} clients from connectivity schedule.")
    print_connectivity_stats(connectivity_schedule, num_clients)
    cfg["model"]["params"]["num_clients"] = num_clients

    cfg["gpu"] = args.gpu
    cfg["seeds"] = args.seeds

    # FedSpace flag overrides agg_mode for convenience
    if args.fedspace:
        effective_mode = "fedspace"
    elif args.full_participation:
        effective_mode = "sync"  # with 'full_table' this behaves as per-slot full aggregation
    else:
        effective_mode = args.agg_mode

    cfg["agg_mode"] = effective_mode
    cfg["buffer_M"] = args.buffer_M

    for sd in args.seeds:
        run_one_seed(cfg, sd, args, connectivity_schedule, num_clients)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--config", default="mnist_minibatch/svfl.yaml")
    ap.add_argument("--gpu",   type=lambda s: [int(s)], required=True)
    ap.add_argument("--seeds", type=int, nargs='+', required=True)
    ap.add_argument("--no_wandb", action="store_true")

    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--fedspace", action="store_true",
                     help="Dynamic FedSpace aggregation schedule")
    grp.add_argument("--full_participation", action="store_true",
                     help="Aggregate every slot with all clients")
    
    ap.add_argument(
    "--agg_mode",
    choices=["sync", "async", "fedbuff", "fedspace"],
    default="sync",
    help="Synchronous, Asynchronous, FedBuff(M), or FedSpace (scheduler).",
    )
    ap.add_argument(
        "--buffer_M",
        type=int,
        default=96,
        help="FedBuff buffer size M (unique clients required before aggregation).",
    )

    main(ap.parse_args())
