"""
array_analysis.py
=================
Pre-generation design report for a microphone-array config (numpy only).

1. Per pair group (INTRA = inside one sub-array, INTER = across sub-arrays):
   distances, tonal-ambiguity frequency c/(2d), lag window, bearing error per
   sample of TDOA error, and how many pairs each `inter_pairs` choice keeps.
2. Localisation accuracy of the sub-array LAYOUT itself (works for 2, 3, ...
   sub-arrays): Monte-Carlo bearing-only triangulation -- every sub-array
   gives a bearing with `doa_std_deg` error, the 3-D position is the
   least-squares intersection of the rays -- for a source at range R and a
   sweep of azimuths. Shows directly where a layout is blind (e.g. two arrays
   when the source lies on the line joining them).

    python array_analysis.py --config example_config.yaml
    python array_analysis.py --config example_config.yaml --doa_std_deg 1.0 --elevation_deg 30
"""
import argparse
import numpy as np
import yaml

import array_geometry as ag


def pair_report(mic_positions, fs, c=343.0):
    names, mat = ag.as_matrix(mic_positions)
    groups = ag.infer_mic_groups(names)
    glist = list(dict.fromkeys(groups))
    print(f"{len(names)} mics in {len(glist)} group(s) {glist}, fs={fs} Hz, c={c} m/s")
    intra, inter = ag.split_pairs_by_group(names, groups, "ref") # "all" - "ref" - k
    for label, pairs in (("INTRA", intra), ("INTER (all pairs)", inter)):
        if not pairs:
            print(f"\n[{label}] no pairs")
            continue
        d = np.array([np.linalg.norm(mat[i] - mat[j]) for i, j in pairs])
        print(f"\n[{label}] {len(pairs)} pairs")
        print(f"  distance          : min {d.min():.3f} m, max {d.max():.3f} m")
        print(f"  tonal-ambiguity f : {c/(2*d.max()):.0f} Hz (longest) ... {c/(2*d.min()):.0f} Hz (shortest)")
        print(f"  max TDOA          : {d.max()/c*1e3:.3f} ms = {d.max()/c*fs:.1f} samples; "
              f"lag window with +20% margin = +/-{1.2*d.max()/c*fs:.0f}")
        print(f"  1 sample of TDOA error -> bearing error ~ {np.degrees(c/fs/d.max()):.2f} deg (longest pair, broadside)")
    if inter:
        print("\nInter-array pairs kept per `inter_pairs` setting (FeatureConfig.inter_pairs):")
        for opt in ("all", 2, "ref"):
            print(f"  {str(opt):>4s}: {len(ag.split_pairs_by_group(names, groups, opt)[1])} pairs")
    return names, mat, groups


def localisation_report(mat, groups, doa_std_deg=2.0, elevation_deg=30.0,
                        ranges=(20.0, 40.0), azimuths=range(0, 360, 30), n=3000, seed=0):
    rng = np.random.default_rng(seed)
    glist = list(dict.fromkeys(groups))
    if len(glist) < 2:
        print("\nOnly one sub-array: no triangulation possible (bearing only).")
        return
    P = np.stack([mat[[i for i, g in enumerate(groups) if g == name]].mean(axis=0) for name in glist])
    K = len(P)
    sig = np.radians(doa_std_deg)
    el = np.radians(elevation_deg)
    print(f"\nLayout accuracy: {K} sub-arrays, bearing std {doa_std_deg} deg each, source elevation {elevation_deg} deg")
    print("centroids: " + ", ".join(f"{g}=({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})" for g, p in zip(glist, P)))
    I = np.eye(3)
    for R in ranges:
        meds, p90s = [], []
        for az in azimuths:
            a = np.radians(az)
            s = R * np.array([np.cos(el) * np.cos(a), np.cos(el) * np.sin(a), np.sin(el)])
            d0 = (s - P) / np.linalg.norm(s - P, axis=1, keepdims=True)           # (K,3)
            noise = sig * rng.standard_normal((n, K, 3))
            noise -= (noise * d0).sum(-1, keepdims=True) * d0                        # perpendicular only
            d = d0 + noise
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
            M = I - d[..., :, None] * d[..., None, :]                                 # (n,K,3,3)
            A = M.sum(1) + 1e-9 * I
            b = np.einsum("nkij,kj->ni", M, P)
            x = np.linalg.solve(A, b[..., None])[..., 0]
            e = np.linalg.norm(x - s, axis=1)
            meds.append(np.median(e)); p90s.append(np.percentile(e, 90))
        print(f"\n  R = {R:.0f} m   3-D position error, median / 90th pct [m]")
        print("  azimuth : " + " ".join(f"{az:>9d}" for az in azimuths))
        print("  error   : " + " ".join(f"{m:4.1f}/{p:4.1f}" for m, p in zip(meds, p90s)))
        print(f"  worst azimuth: median {max(meds):.1f} m, 90th pct {max(p90s):.1f} m")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--doa_std_deg", type=float, default=2.0)
    ap.add_argument("--elevation_deg", type=float, default=30.0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    names, mat, groups = pair_report(ag.from_config(cfg["array"]), cfg["fs"])
    localisation_report(mat, groups, a.doa_std_deg, a.elevation_deg)
