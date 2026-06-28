#!/usr/bin/env python3
"""
Plot the free-surface height of a swe_noise run on a lon-lat map.

snapy writes the six cube faces unrolled into a single array, with 2D `lon` and
`lat` coordinate variables, so we simply scatter every cell at its (lon, lat).
The shallow-water prognostic stored as `rho` is the geopotential gh; height is
h = rho / g.

    python plot_height.py out_noise/swe_noise.out0.00000.nc -o height.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

G = 9.80616


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ncfile", help="NetCDF output written by swe_noise.py")
    p.add_argument("-o", "--out", default="height.png", help="output PNG")
    p.add_argument("-t", "--time", type=int, default=-1,
                   help="time index to plot (default: last)")
    args = p.parse_args()

    d = Dataset(args.ncfile)
    lon = np.asarray(d["lon"][0]) * 180.0 / np.pi      # (nx3, nx2), degrees
    lat = np.asarray(d["lat"][0]) * 180.0 / np.pi
    h = np.asarray(d["rho"][args.time, 0]) / G          # (nx3, nx2), metres
    t_days = float(d["time"][args.time]) / 86400.0

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sc = ax.scatter(lon.ravel(), lat.ravel(), c=h.ravel(),
                    s=4, cmap="RdBu_r", marker="s", linewidths=0)
    ax.set_xlim(0, 360); ax.set_ylim(-90, 90)
    ax.set_xlabel("longitude (deg)"); ax.set_ylabel("latitude (deg)")
    ax.set_title(f"shallow-water free-surface height @ day {t_days:.2f}")
    fig.colorbar(sc, ax=ax, label="height (m)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}  (h range {h.min():.1f}-{h.max():.1f} m)")


if __name__ == "__main__":
    main()
