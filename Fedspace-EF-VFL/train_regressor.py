#!/usr/bin/env python
# train_fedspace_regressor.py   ❷  ─────────────────────────────────────
import numpy as np, pathlib, joblib
from sklearn.ensemble import RandomForestRegressor
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument("--data", default="datasets/fedspace_offline.npz")
ap.add_argument("--out",  default="models/utility_regressor.pkl")
args = ap.parse_args()

ds   = np.load(args.data)
X    = np.hstack([ds["S"], ds["T"]])     # (N, K+1)
y    = ds["dF"].ravel()                  # Δf

rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        max_features="sqrt",
        random_state=0,
        n_jobs=-1)
rf.fit(X, y)

pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
joblib.dump(rf, args.out)
print("🔮  saved regressor to", args.out,
      "|  training samples:", len(y))
