````markdown
# README

Train and evaluate split-VFL variants (SVFL / EFVFL / CVFL) under different aggregation regimes (Sync / Async / FedBuff / FedSpace) with deterministic connectivity.

---

## Quick start

1) **Install deps (Python ≥ 3.10):**
```bash
pip install -U torch pytorch-lightning torchmetrics wandb pyyaml numpy requests scikit-learn joblib
````

2. **Run one of the examples below** (assumes `configs/` and the connectivity file referenced in the YAML exist).

---

## Examples

### FedSpace (dynamic schedule over deterministic connectivity)

> ⚠️ Requires a trained *utility regressor*. See **“Prep for FedSpace (utility regressor)”** below.

```bash
python main.py --config mnist_minibatch/svfl.yaml --gpu 0 --seeds 0 --no_wandb --agg_mode fedspace
```

### FedBuff with buffer size M=96

```bash
python main.py --config mnist_minibatch/efvfl_0.2k.yaml --gpu 0 --seeds 0 --no_wandb --agg_mode fedbuff --buffer_M 96
```

### Synchronous (per-slot aggregation)

```bash
python main.py --config mnist_minibatch/cvfl_0.05k.yaml --gpu 0 --seeds 0 --no_wandb --agg_mode sync
```

> Tip: to sweep seeds, pass multiple integers: `--seeds 0 1 2`.

---

## CLI reference

```
--config              Path under configs/ to a YAML (e.g., mnist_minibatch/svfl.yaml)
--gpu                 Single GPU index (e.g., 0). Internally parsed as [0].
--seeds               One or more integers (e.g., 0 1 2)
--no_wandb            Disable Weights & Biases logging (default is enabled)

--agg_mode            {sync, async, fedbuff, fedspace}
--buffer_M            FedBuff buffer size M (unique clients required). Default: 96

--fedspace            Force FedSpace scheduling (convenience flag; equivalent to --agg_mode fedspace)
--full_participation  Aggregate every slot with all clients (overrides partial participation)
```

### Aggregation modes

* `sync`  All updates in the credited set are aggregated when the slot triggers.
* `async` Aggregate on arrivals in the **current** slot only (no union buffer).
* `fedbuff` Aggregate when the union buffer reaches `M` unique clients (`--buffer_M`).
* `fedspace` Dynamic schedule chosen by a trained utility regressor over a time window.

---

## Prep for FedSpace (utility regressor)

FedSpace uses a learned utility regressor to decide **when** to aggregate. To enable `--agg_mode fedspace`, do this one-time setup:

### 1) Dry runs **without** the FedSpace scheduler

Run a few training jobs using any non-FedSpace mode (e.g., `sync`, `async` or `fedbuff`) so the code logs per-round **staleness** and **loss**. Use different configs/seeds to enrich the dataset.

Examples:

```bash
# SVFL baseline, sync aggregation, 3 seeds
python main.py --config mnist_minibatch/svfl.yaml       --gpu 0 --seeds 0 1 2 --no_wandb --agg_mode sync

# EFVFL with FedBuff(M=96), 2 seeds
python main.py --config mnist_minibatch/efvfl_0.2k.yaml --gpu 0 --seeds 0 1   --no_wandb --agg_mode fedbuff --buffer_M 96
```

These runs write arrays per run (e.g., `staleness.npy`, `loss_round.npy`) under `results/<experiment>-s<seed>/`.

### 2) Build the offline dataset for the regressor

```bash
python build_dataset.py --results_dir results --out datasets/fedspace_offline.npz
```

### 3) Train the utility regressor

```bash
python train_regressor.py --data datasets/fedspace_offline.npz --out models/utility_regressor.pkl
```

After step 3, `models/utility_regressor.pkl` exists and FedSpace scheduling can be used.

### 4) Run with FedSpace scheduling

```bash
python main.py --config mnist_minibatch/svfl.yaml --gpu 0 --seeds 0 --no_wandb --agg_mode fedspace
```

---

## Connectivity

On startup, the program loads a deterministic connectivity table from:

```yaml
connectivity:
  path: <your/connectivity_file>
```

It prints stats (e.g., average number of online clients per slot, coverage). FedSpace uses this table with the trained regressor to plan aggregation slots.

---

## Logging & outputs

* **Weights & Biases**: enabled by default. Use `--no_wandb` to disable.
* **Offline artifacts** (when `--no_wandb`): saved under

  ```
  results/<experiment-name>-s<seed>/
  ```

  plus debug arrays (staleness, bytes-per-round, aggregation mask) saved on train end.

---

## Troubleshooting

* **Missing regressor**: `--agg_mode fedspace` requires `models/utility_regressor.pkl`. If absent, do the three prep steps above.
* **GPU arg**: pass a single index (`--gpu 0`). Multi-GPU via this CLI isn’t wired up.
* **W&B**: ensure `wandb login` is configured, or add `--no_wandb`.
* **Buffer size**: `--buffer_M` only applies when `--agg_mode fedbuff`.