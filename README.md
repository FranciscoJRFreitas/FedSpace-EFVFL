# FedSpace-EFVFL

This repository contains code and experiment outputs for **FedSpace-EFVFL**, which combines ideas from scheduling in space-aware FL (FedSpace) with communication-efficient Error-Feedback Vertical Federated Learning (EFVFL).

## Upstream basis

The implementation in the root folder **`Fedspace-EFVFL/`** is **based on** the EFVFL reference repository:

> https://github.com/Valdeira/EF-VFL

## Contributions

The contributions referenced in the paper are implemented in the following files inside `Fedspace-EFVFL/`:

- `build_dataset.py`
- `train_regressor.py`
- `lightning_splitnn.py`
- `mnist_model.py`
- `connectivity.py`
- `scheduler.py`

These changes add the functionality specific to the FedSpace-EFVFL integration and the experiments reported in the manuscript.

---

## Experiments & results

- All the **outputs from my runs** (summaries, metrics) are stored under the root **`Experiments/`** folder.
- To **visualize results and reproduce figures by experiment**, open:
  - `reproduce_figs_by_experiment.ipynb`
- To **visualize connectivity** (e.g., contact windows/schedules), open:
  - `connectivity_visualization.ipynb`

> Both notebooks expect the results to be present under `Experiments/` (and/or the default paths referenced inside).

---

## Quick start (notebooks)

1. Create and activate a Python environment (Python ≥ 3.9 recommended).
2. Install dependencies (typical stack):

```bash
   pip install numpy matplotlib pytorch-lightning torch torchvision jupyter
```
3. Launch Jupyter and open the notebooks:

   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

## Acknowledgments

* This work builds upon the EFVFL codebase and method by Valdeira *et al.*
  Source: [https://github.com/Valdeira/EF-VFL](https://github.com/Valdeira/EF-VFL)

If you use this repository, please also acknowledge the EFVFL authors and cite their work accordingly.