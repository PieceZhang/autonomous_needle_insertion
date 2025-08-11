"""
Register two 3D point sets with unknown correspondences where the MRI set
contains 9 *meta-fiducials*, each built from two visible markers (18 points total).

Steps:
A) MRI pairing: pair 18 points into 9 meta-fiducials via minimum-sum perfect matching
   on the complete graph (Blossom). Optionally constrain edges around a known pair distance.
B) Collapse each pair to its midpoint (virtual fiducial center).
C) Solve unlabeled 9↔9 correspondences by matching pairwise-distance signatures (Hungarian).
D) Estimate rigid (or similarity) transform via Kabsch/Umeyama (SVD).

# Known intra-meta spacing (recommended if you know it)
python register_meta_fids.py mri_18.tsv sensor_9.tsv \
  --pair-dist 12.5 --pair-tol 1.0 \
  --save-tsv matched.tsv --out-npz reg_meta.npz

# If spacing unknown (works well when each pair is much closer than inter-fiducial gaps)
python register_meta_fids.py mri_18.tsv sensor_9.tsv \
  --save-tsv matched.tsv --out-npz reg_meta.npz

References:
- Max-/min-weight matching (Blossom): NetworkX docs. 
- Hungarian algorithm (SciPy): linear_sum_assignment.
- Kabsch / Umeyama: rigid/similarity least-squares alignment with SVD.
- Euclidean distance matrices & rigid-motion invariance.

"""

import argparse
import os
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment

# ---------------- I/O ----------------
def load_points_table(path: str) -> np.ndarray:
    """Load N x 3 points from a TSV/CSV (header optional).
    Inference by extension: .tsv/.tab -> "\t", .csv -> ","; otherwise let pandas infer.
    Prefers columns named x,y,z (case-insensitive); else uses the first three numeric columns.
    """
    # Infer delimiter from extension (case-insensitive)
    ext = os.path.splitext(path)[1].lower()
    if ext in {".tsv", ".tab"}:
        sep = "\t"
    elif ext == ".csv":
        sep = ","
    else:
        sep = None  # let pandas infer

    try:
        import pandas as pd
        df = pd.read_csv(path, sep=sep, comment="#", engine="python")
        lower = {c.lower(): c for c in df.columns}
        if {"x", "y", "z"} <= set(lower):
            arr = df[[lower["x"], lower["y"], lower["z"]]].to_numpy(dtype=float)
        else:
            num = df.select_dtypes(include=[np.number])
            if num.shape[1] < 3:
                raise ValueError("Less than 3 numeric columns.")
            arr = num.iloc[:, :3].to_numpy(dtype=float)
    except Exception:
        # Fallbacks with numpy; try inferred sep first, then common alternatives
        candidates = []
        if sep is not None:
            candidates.append(sep)
        # try both tab and comma as backups
        for d in ("\t", ","):
            if d not in candidates:
                candidates.append(d)
        last_err = None
        for d in candidates:
            try:
                arr = np.loadtxt(path, delimiter=d, dtype=float)
                arr = np.atleast_2d(arr)
                if arr.shape[1] >= 3:
                    arr = arr[:, :3]
                    break
            except Exception as e:
                last_err = e
                arr = None
        if arr is None:
            raise last_err if last_err is not None else ValueError(f"Failed to parse {path}")
    if arr.shape[0] < 3:
        raise ValueError("Need at least 3 non-collinear points.")
    return arr

# ------------- Utilities -------------
def pairwise_distances(X: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

def distance_signatures(D: np.ndarray) -> np.ndarray:
    N = D.shape[0]
    S = np.empty((N, N-1), dtype=float)
    for i in range(N):
        row = np.delete(D[i], i)
        row.sort()
        S[i] = row
    return S

def match_unlabeled(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Match B to A by Hungarian on their distance-signature L2 costs."""
    Da, Db = pairwise_distances(A), pairwise_distances(B)
    Sa, Sb = distance_signatures(Da), distance_signatures(Db)
    C = np.linalg.norm(Sa[:, None, :] - Sb[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(C)  # Hungarian
    perm = np.empty(A.shape[0], dtype=int)
    perm[rows] = cols
    return perm, C

def rigid_transform_row(src: np.ndarray, dst: np.ndarray):
    """Solve R,t for dst ≈ src@R + t with det(R)=+1 (Kabsch)."""
    c_src, c_dst = src.mean(axis=0), dst.mean(axis=0)
    X, Y = src - c_src, dst - c_dst
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = c_dst - c_src @ R
    return R, t

def similarity_transform_row(src: np.ndarray, dst: np.ndarray):
    """Solve s,R,t for dst ≈ s*(src@R) + t (Umeyama)."""
    c_src, c_dst = src.mean(axis=0), dst.mean(axis=0)
    X, Y = src - c_src, dst - c_dst
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        S[-1] *= -1
        R = U @ Vt
    s = S.sum() / (X**2).sum()
    t = c_dst - s*(c_src @ R)
    return s, R, t

def rmse(A: np.ndarray, B: np.ndarray) -> float:
    return np.sqrt(np.mean(np.sum((A - B)**2, axis=1)))

# ------------- MRI pairing -------------
def pair_meta_fiducials(points18: np.ndarray,
                        known_pair_dist: float=None,
                        tol: float=None,
                        max_relax_steps: int=3):
    """
    Pair 18 MRI points into 9 disjoint pairs.
    If known_pair_dist and tol are given, only allow edges within |d - known_pair_dist| <= tol.
    Otherwise, do unconstrained minimum-sum perfect matching (by maximizing negative distance).

    Returns:
        pairs: list of (i,j) index tuples
        pair_dists: np.array of distances for each pair
    """
    n = points18.shape[0]
    if n != 18:
        raise ValueError(f"Expected 18 MRI points for meta-fiducials; got {n}.")
    D = pairwise_distances(points18)

    def try_match(allow_mask):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i+1, n):
                if allow_mask[i, j]:
                    # maximize weight -> use negative distance for min-sum pairing
                    G.add_edge(i, j, weight=-float(D[i, j]))
        M = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
        if len(M)*2 == n:
            pairs = sorted([(min(i,j), max(i,j)) for i,j in M])
            dists = np.array([D[i, j] for (i, j) in pairs], dtype=float)
            return pairs, dists
        return None, None

    # Build initial allow_mask
    allow = np.ones((n, n), dtype=bool)
    allow[np.arange(n), np.arange(n)] = False
    allow = np.triu(allow, k=1)

    if known_pair_dist is not None and tol is not None and tol > 0:
        # Start with tight band around known distance; relax if needed.
        for step in range(max_relax_steps+1):
            band = tol * (2**step)
            mask = allow & (np.abs(D - known_pair_dist) <= band)
            pairs, dists = try_match(mask)
            if pairs is not None:
                return pairs, dists
        # Fallback: unconstrained
        pairs, dists = try_match(allow)
        if pairs is not None:
            return pairs, dists
        raise RuntimeError("Failed to find a perfect pairing even after relaxation.")
    else:
        # Unconstrained minimum-sum pairing
        pairs, dists = try_match(allow)
        if pairs is None:
            raise RuntimeError("Failed to compute a perfect pairing.")
        return pairs, dists

# ------------- Main pipeline -------------
def main():
    ap = argparse.ArgumentParser(description="Register 3D point sets with meta-fiducial pairing on MRI.")
    ap.add_argument("mri_file", help="Path to 18-point MRI marker file (TSV or CSV; two per meta-fiducial).")
    ap.add_argument("sensor_file", help="Path to 9-point sensor marker file (TSV or CSV; one per meta-fiducial).")
    ap.add_argument("--pair-dist", type=float, default=None,
                    help="Known distance between the two markers of a meta-fiducial (in MRI units).")
    ap.add_argument("--pair-tol", type=float, default=None,
                    help="Tolerance for --pair-dist. If set, pairing will be constrained and relaxed if needed.")
    ap.add_argument("--similarity", action="store_true",
                    help="Estimate isotropic scale in addition to rigid transform (Umeyama).")
    ap.add_argument("--save-tsv", default=None,
                    help="Optional path to save correspondences and residuals (TSV).")
    ap.add_argument("--out-npz", default="results/registration_result.npz",
                    help="Path to save transform & artifacts.")
    args = ap.parse_args()

    # Load
    mri_raw = load_points_table(args.mri_file)   # expect 18 x 3
    sensor9 = load_points_table(args.sensor_file)  # expect 9 x 3
    if mri_raw.shape[0] != 18 or mri_raw.shape[1] != 3:
        raise ValueError("MRI file must be 18 x 3 (two per meta-fiducial).")
    if sensor9.shape[0] != 9 or sensor9.shape[1] != 3:
        raise ValueError("Sensor file must be 9 x 3 (one per meta-fiducial).")

    # A) Pair the 18 MRI points into 9 meta-fiducials
    pairs, pair_dists = pair_meta_fiducials(
        mri_raw,
        known_pair_dist=args.pair_dist,
        tol=args.pair_tol if args.pair_tol is not None else None
    )
    # centers: midpoint of each pair
    mri9 = np.vstack([0.5*(mri_raw[i] + mri_raw[j]) for (i, j) in pairs])

    # B) Unlabeled 9↔9 matching by distance-signature + Hungarian
    perm, cost = match_unlabeled(mri9, sensor9)
    sensor9_perm = sensor9[perm]

    # C) Estimate transform (sensor -> MRI)
    if args.similarity:
        s, R, t = similarity_transform_row(sensor9_perm, mri9)
        aligned = s*(sensor9_perm @ R) + t
    else:
        R, t = rigid_transform_row(sensor9_perm, mri9)
        s = 1.0
        aligned = sensor9_perm @ R + t

    # D) Error
    err = rmse(mri9, aligned)

    # ---- Report ----
    np.set_printoptions(precision=6, suppress=True)
    print("Meta-fiducial pairing (MRI):")
    for k, (i, j) in enumerate(pairs):
        print(f"  pair {k:02d}: ({i},{j})  dist={pair_dists[k]:.4f}")
    print(f"Pair dist mean±sd: {pair_dists.mean():.4f} ± {pair_dists.std(ddof=1):.4f}")

    print("\nHungarian correspondences (sensor index -> MRI meta index):")
    print(perm)
    print("\nEstimated transform (sensor -> MRI):")
    print(f"Scale s: {s:.8f}  (≈1.0 for rigid)")
    print("Rotation R:\n", R)
    print("Translation t:\n", t)
    print(f"RMSE after alignment: {err:.6f}")

    # Save artifacts
    np.savez(
        args.out_npz,
        R=R, t=t, s=s, rmse=err,
        perm=perm,
        pairs=np.array(pairs, dtype=int),
        pair_dists=pair_dists,
        mri_raw=mri_raw,
        mri_centers=mri9,
        sensor=sensor9,
        sensor_permuted=sensor9_perm,
        sensor_aligned=aligned,
        cost_matrix=cost
    )
    print(f"\nSaved transform & artifacts to: {args.out_npz}")

    if args.save_tsv:
        res = np.linalg.norm(mri9 - aligned, axis=1)
        header = "mri_meta_idx\tsensor_idx\tx_mri\ty_mri\tz_mri\tx_sensor\ty_sensor\tz_sensor\tx_aligned\ty_aligned\tz_aligned\tresidual"
        rows = []
        for i in range(9):
            rows.append([
                i, int(perm[i]),
                *mri9[i].tolist(),
                *sensor9_perm[i].tolist(),
                *aligned[i].tolist(),
                float(res[i])
            ])
        np.savetxt(args.save_tsv, np.array(rows, dtype=float), delimiter="\t", header=header, comments="")
        print(f"Saved correspondences/residuals TSV to: {args.save_tsv}")

if __name__ == "__main__":
    main()