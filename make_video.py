#!/usr/bin/env python3
r"""
Turn a jupiter_run.py output directory into lat-lon videos of relative
vorticity (and, for the moist run, water-vapour mixing ratio q).

snapy writes one combined NetCDF per frame: the six cube faces unrolled into a
(nx3, nx2) array with 2D `lon`/`lat` and contravariant velocities `vel2,vel3`.
We:
  1. assign every cell to its cube face (where its (alpha,beta) is in [-pi/4,pi/4]),
  2. rotate (vel2,vel3) -> geographic (east,north) with paddle's validated map,
  3. interpolate east/north (and q) onto a regular lon-lat grid (triangulation
     built once, reused for all frames),
  4. compute relative vorticity  zeta = (1/(a cos lat)) [d v/d lon - d(u cos lat)/d lat],
  5. render one frame per file and encode an mp4 with ffmpeg (imageio).

    python make_video.py /data/tmp/jupiter_moist --moist -o videos/jupiter_moist
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.spatial import Delaunay
import imageio.v2 as imageio

import paddle.cubed_sphere_remap as csr

G = 24.79
A = 71492000.0          # Jupiter radius (m)
QUARTER = np.pi / 4 + 1e-6


def assign_faces(lon, lat):
    """Per cell: face id and local (alpha,beta). lon,lat flat (radians)."""
    face = np.full(lon.shape, -1, dtype=np.int64)
    alpha = np.zeros_like(lon); beta = np.zeros_like(lon)
    for f in range(6):
        a, b = csr.lonlat_to_face_ab(f, lon, lat)
        inside = (np.abs(a) <= QUARTER) & (np.abs(b) <= QUARTER) & (face < 0)
        face[inside] = f; alpha[inside] = a[inside]; beta[inside] = b[inside]
    return face, alpha, beta


def to_east_north(face, alpha, beta, lon, lat, v2, v3):
    """Contravariant (vel2,vel3) -> geographic (east, north)."""
    east = np.zeros_like(v2); north = np.zeros_like(v2)
    z = np.zeros_like(v2)
    for f in range(6):
        m = face == f
        gx, gy, gz = csr._local_contra_to_global_xyz(
            f, z[m], v2[m], v3[m], alpha[m], beta[m])
        lo, la = lon[m], lat[m]
        east[m] = -np.sin(lo) * gx + np.cos(lo) * gy
        north[m] = (-np.sin(la) * np.cos(lo) * gx
                    - np.sin(la) * np.sin(lo) * gy + np.cos(la) * gz)
    return east, north


def build_interp(lon, lat, nlon, nlat):
    """Precompute barycentric interpolation weights from the (static) cube grid
    to a regular lon-lat grid, so each frame is a fast gather (no re-triangulation).

    Returns a dict with the seam mask, target vertex indices (M,3) and weights
    (M,3), the valid mask, and the grid axes.
    """
    seam = (lon < 0.4) | (lon > 2 * np.pi - 0.4)
    pl = np.concatenate([lon, lon[seam] + 2 * np.pi, lon[seam] - 2 * np.pi])
    pa = np.concatenate([lat, lat[seam], lat[seam]])
    tri = Delaunay(np.column_stack([pl, pa]))

    glon = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
    glat = np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, nlat)
    LO, LA = np.meshgrid(glon, glat)
    targets = np.column_stack([LO.ravel(), LA.ravel()])

    simplex = tri.find_simplex(targets)
    valid = simplex >= 0
    T = tri.transform[simplex[valid]]
    X = targets[valid] - T[:, 2, :]
    bary = np.einsum("nij,nj->ni", T[:, :2, :], X)
    w = np.c_[bary, 1.0 - bary.sum(axis=1)]            # (Mvalid, 3)
    verts = tri.simplices[simplex[valid]]               # (Mvalid, 3)
    return {"seam": seam, "valid": valid, "verts": verts, "w": w,
            "shape": LO.shape, "glon": glon, "glat": glat}


def interp(P, vals):
    """Apply precomputed weights to a flat per-cell field -> lon-lat grid."""
    pv = np.concatenate([vals, vals[P["seam"]], vals[P["seam"]]])
    out = np.full(P["valid"].size, np.nan)
    out[P["valid"]] = (P["w"] * pv[P["verts"]]).sum(axis=1)
    return out.reshape(P["shape"])


def vorticity(u, v, glon, glat):
    """Relative vorticity on a regular lon-lat grid (1/s)."""
    coslat = np.cos(glat)[:, None]
    dlon = glon[1] - glon[0]
    dlat = glat[1] - glat[0]
    dv_dlon = np.gradient(v, dlon, axis=1)
    ducos_dlat = np.gradient(u * coslat, dlat, axis=0)
    return (dv_dlon - ducos_dlat) / (A * coslat)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("indir", help="jupiter_run.py output dir")
    p.add_argument("--moist", action="store_true", help="also render moisture q")
    p.add_argument("-o", "--out", default=None, help="output basename (default <indir>/video)")
    p.add_argument("--nlon", type=int, default=720)
    p.add_argument("--nlat", type=int, default=360)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--vmax", type=float, default=None, help="vorticity colour limit (1/s)")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "*.nc")))
    if not files:
        raise SystemExit(f"no .nc files in {args.indir}")
    out = args.out or os.path.join(args.indir, "video")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # --- static setup from frame 0 ---
    d0 = Dataset(files[0])
    lon = np.asarray(d0["lon"][0]).ravel()
    lat = np.asarray(d0["lat"][0]).ravel()
    shape = np.asarray(d0["lon"][0]).shape
    face, alpha, beta = assign_faces(lon, lat)
    P = build_interp(lon, lat, args.nlon, args.nlat)
    glon, glat = P["glon"], P["glat"]
    lon_deg, lat_deg = np.degrees(glon), np.degrees(glat)

    # auto colour scale for vorticity from a mid/late frame
    if args.vmax is None:
        dm = Dataset(files[len(files) // 2])
        u, v = to_east_north(face, alpha, beta, lon, lat,
                             np.asarray(dm["vel2"][0, 0]).ravel(),
                             np.asarray(dm["vel3"][0, 0]).ravel())
        z = vorticity(interp(P, u), interp(P, v), glon, glat)
        args.vmax = float(np.nanpercentile(np.abs(z), 99))
    vmax = args.vmax

    writers = {"vort": imageio.get_writer(out + "_vorticity.mp4", fps=args.fps,
                                          codec="libx264", quality=8)}
    if args.moist:
        writers["q"] = imageio.get_writer(out + "_moisture.mp4", fps=args.fps,
                                          codec="libx264", quality=8)

    for i, fn in enumerate(files):
        d = Dataset(fn)
        t_day = float(d["time"][0]) / 86400.0
        v2 = np.asarray(d["vel2"][0, 0]).ravel()
        v3 = np.asarray(d["vel3"][0, 0]).ravel()
        u, v = to_east_north(face, alpha, beta, lon, lat, v2, v3)
        U = interp(P, u); V = interp(P, v)
        zeta = vorticity(U, V, glon, glat)

        fig, ax = plt.subplots(figsize=(11, 5.2))
        im = ax.pcolormesh(lon_deg, lat_deg, zeta, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_title(f"relative vorticity  (day {t_day:.0f})")
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        fig.colorbar(im, ax=ax, label="$\\zeta$ (s$^{-1}$)")
        fig.tight_layout()
        fig.canvas.draw()
        writers["vort"].append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)

        if args.moist and "r_qv" in d.variables:
            q = interp(P, np.asarray(d["r_qv"][0, 0]).ravel())
            fig, ax = plt.subplots(figsize=(11, 5.2))
            im = ax.pcolormesh(lon_deg, lat_deg, q, cmap="viridis", shading="auto")
            ax.set_title(f"water-vapour mixing ratio q  (day {t_day:.0f})")
            ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
            fig.colorbar(im, ax=ax, label="q")
            fig.tight_layout(); fig.canvas.draw()
            writers["q"].append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            plt.close(fig)

        if i % 50 == 0:
            print(f"frame {i+1}/{len(files)} (day {t_day:.0f})", flush=True)

    for w in writers.values():
        w.close()
    print("wrote", out + "_vorticity.mp4" + (" and _moisture.mp4" if args.moist else ""))


if __name__ == "__main__":
    main()
