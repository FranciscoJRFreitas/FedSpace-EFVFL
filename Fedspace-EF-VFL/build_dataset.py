import numpy as np, glob, argparse, pathlib

TOL = 1e-7                         # two losses equal if |delta| < TOL

def compress_loss_series(loss_raw, tol=1e-7):
    """Return an array with the first value of every constant run
       *plus* the very last value so len == #rounds + 1."""
    loss_raw = loss_raw.ravel()
    changepoints = np.concatenate((
        [True],
        np.abs(np.diff(loss_raw)) > tol
    ))
    out = loss_raw[changepoints]

    if out[-1] != loss_raw[-1]:
        # append the final loss to ensure len == R+1
        out = np.append(out, loss_raw[-1])

    return out

def main(results_dir="results",
         out="datasets/fedspace_offline.npz"):

    triples = []     # list of (staleness_row, T_before, delta_f)

    for run in glob.glob(f"{results_dir}/*/"):
        r = pathlib.Path(run)
        s_path, f_path = r / "staleness.npy", r / "loss_round.npy"
        if not (s_path.exists() and f_path.exists()):
            print(" - skipping", r.name, "(missing files)")
            continue

        S_mat = np.load(s_path)          # shape (R, K)
        f_raw = np.load(f_path)          # long 1‑D array

        f = compress_loss_series(f_raw)  # one loss per round (+1)

        if len(f) == len(S_mat):          # we're missing the final point only
            f = np.append(f, f[-1])       # pad with duplicate tail value
        if len(f) != len(S_mat) + 1:      # anything else -> real mismatch
            raise ValueError(
                f"{r.name}: after compression got len(f)={len(f)}, "
                f"but need len(S)+1 = {len(S_mat)+1}. "
                "Check your logging logic or adjust TOL."
            )

        for i in range(len(S_mat)):      # i = 0 … R-1
            triples.append((S_mat[i], f[i], f[i] - f[i+1]))

    if not triples:
        raise RuntimeError("no runs found with the required .npy files")

    S  = np.stack([t[0] for t in triples])            # (N, K)
    T  = np.stack([t[1] for t in triples]).reshape(-1, 1)
    dF = np.stack([t[2] for t in triples]).reshape(-1, 1)

    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, S=S.astype(np.int16),
                             T=T.astype(np.float32),
                             dF=dF.astype(np.float32))
    print("✅  wrote", pathlib.Path(out).resolve(),
          " |  samples:", len(triples))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="datasets/fedspace_offline.npz")
    main(**vars(ap.parse_args()))