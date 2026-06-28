#!/usr/bin/env python3
"""
Global shallow-water model on the gnomonic-equiangle cubed sphere, initialized
from a resting flat free surface plus random noise of a configurable scale.

Built on snapy (https://github.com/chengcli/snapy):
  * `shallow-water` equation of state,
  * `shallow-roe` Riemann solver,
  * `coriolis` forcing on the rotating sphere,
six cube faces held on a single process (`blocks_per_process: 6`).

Unlike the bundled Williamson-1992 (W92) benchmarks, the initial state here is
trivial to set -- a flat ocean at rest -- so no contravariant<->geographic wind
rotation (and hence no `paddle`) is needed. We then perturb the free-surface
height field with band-limited Gaussian noise:

    h(face, j, k) = H0 + dh,      dh ~ amplitude * N(0, 1) (optionally smoothed)
    gh             = G * h        (snapy's shallow-water prognostic is geopotential)
    velocities     = 0

The Coriolis force drives geostrophic adjustment of this unbalanced height
field, radiating gravity waves and spinning up freely-evolving shallow-water
turbulence. The two knobs that set "the scale" of the injected noise are:

  --noise-amp     standard deviation of the height perturbation, in metres
  --noise-smooth  number of per-face 3x3 smoothing passes; larger -> longer
                  horizontal correlation length (bigger eddies), smaller std

Run (single process, all six faces, CPU):

    torchrun --nproc_per_node=1 swe_noise.py --device cpu --output-dir out_noise

See README.md for details and multi-GPU notes.
"""
import argparse
import os

import numpy as np
import torch
import yaml
from snapy import Mesh, MeshOptions

# snapy CS_FACE_NAMES, in face-major (block) order: block f sits on FACE_NAMES[f].
FACE_NAMES = ["+X", "+Y", "-X", "+Z", "-Y", "-Z"]

# Background / physical constants (Earth-like, shared with the W92 example).
G = 9.80616          # gravity (m/s^2)
H0 = 8000.0          # mean fluid depth / free-surface height (m)


def ab_to_lonlat(face, alpha, beta):
    """snapy cs_ab_to_lonlat: equiangular (xi=alpha, eta=beta) -> (lon, lat).

    Only used for diagnostics / optional latitude-dependent shaping of the
    noise; the resting initial state itself does not depend on (lon, lat).
    """
    x = np.tan(alpha)
    y = np.tan(beta)
    r = np.sqrt(x * x + y * y + 1.0)
    if face == "+X":
        lon = alpha.copy(); lat = np.arctan(y / np.sqrt(1 + x * x))
    elif face == "+Y":
        lon = alpha + 0.5 * np.pi; lat = np.arctan(y / np.sqrt(1 + x * x))
    elif face == "-X":
        lon = alpha + np.pi; lat = np.arctan(y / np.sqrt(1 + x * x))
    elif face == "-Y":
        lon = alpha + 1.5 * np.pi; lat = np.arctan(y / np.sqrt(1 + x * x))
    elif face == "+Z":
        lon = np.arctan2(x, -y); lat = np.arcsin(1.0 / r)
    elif face == "-Z":
        lon = np.arctan2(x, y); lat = -np.arcsin(1.0 / r)
    else:
        raise ValueError(face)
    lon = np.where(lon < 0.0, lon + 2 * np.pi, lon)
    return lon, lat


def smooth(field, passes):
    """A few in-place 3x3 box-filter passes on a single face (nc3, nc2).

    Each pass roughly doubles the correlation length and shrinks the standard
    deviation, so `passes` is a cheap knob for the *spatial* scale of the noise.
    Edges use replicate padding; cross-face continuity is intentionally left to
    the solver's own panel exchange once the run starts.
    """
    for _ in range(passes):
        p = np.pad(field, 1, mode="edge")
        field = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
        ) / 9.0
    return field


def noisy_height(face, alpha, beta, rng, amp, passes):
    """Free-surface height H0 + band-limited Gaussian noise on one face."""
    nc3, nc2 = alpha.shape
    dh = rng.standard_normal((nc3, nc2))
    if passes > 0:
        dh = smooth(dh, passes)
        # restore unit variance lost to smoothing, so --noise-amp stays metres
        std = dh.std()
        if std > 0:
            dh = dh / std
    return H0 + amp * dh


def run(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ---- distributed: only needed when launched with WORLD_SIZE > 1 ----------
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        import torch.distributed as dist
        import torch.distributed.distributed_c10d as dist_c10d
        import snapy
        backend = config.get("distribute", {}).get("backend", "gloo")
        dist.init_process_group(backend=backend, init_method="env://")
        snapy.distributed.set_process_group(dist_c10d._get_default_group())

    if args.device == "cuda" and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # ---- build the mesh ------------------------------------------------------
    opt = MeshOptions.from_yaml(args.config)
    opt.block().output_dir(args.output_dir)
    if args.nlim is not None:
        opt.block().intg().nlim(args.nlim)
    os.makedirs(args.output_dir, exist_ok=True)
    mesh = Mesh(opt)
    mesh.to(device)

    # ---- initial condition: resting flat surface + random noise --------------
    block_vars = []
    for f, block in enumerate(mesh.blocks):
        coord = block.module("coord")
        x2v = coord.buffer("x2v").cpu().numpy()   # xi  (nc2,)
        x3v = coord.buffer("x3v").cpu().numpy()   # eta (nc3,)
        nc2, nc3 = x2v.size, x3v.size
        alpha, beta = np.meshgrid(x2v, x3v)       # (nc3, nc2): [k, j]

        # independent, reproducible noise per face
        rng = np.random.default_rng(args.seed + 1000 * f)
        h = noisy_height(FACE_NAMES[f], alpha, beta, rng,
                         args.noise_amp, args.noise_smooth)

        w = torch.zeros((4, nc3, nc2, 1), dtype=torch.float64)
        w[0, :, :, 0] = torch.from_numpy(G * h)   # geopotential gh
        w[1, :, :, 0] = 0.0                        # vel1 (radial)
        w[2, :, :, 0] = 0.0                        # vel2 (xi, contravariant)
        w[3, :, :, 0] = 0.0                        # vel3 (eta, contravariant)
        block_vars.append({"hydro_w": w.to(device)})

    block_vars, current_time = mesh.initialize(block_vars)

    # ---- time integration ----------------------------------------------------
    intg = mesh.module("block0.intg")
    cycle = 0
    mesh.make_outputs(block_vars, current_time)
    while not intg.stop(cycle, current_time):
        cycle += 1
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(block_vars)
        mesh.print_cycle_info(block_vars, current_time, dt)
        for stage in range(len(intg.stages)):
            mesh.forward(block_vars, dt, stage)
        err = mesh.check_redo(block_vars)
        if err > 0:
            continue
        if err < 0:
            break
        current_time += dt
        mesh.make_outputs(block_vars, current_time)
    mesh.finalize(block_vars, current_time)

    if world_size > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config",
                   default=os.path.join(os.path.dirname(__file__), "swe_noise.yaml"),
                   help="YAML config (default: swe_noise.yaml next to this script)")
    p.add_argument("--output-dir", default="out_noise",
                   help="directory for NetCDF output (default: out_noise)")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--noise-amp", type=float, default=50.0,
                   help="std-dev of the height perturbation, in metres (default: 50)")
    p.add_argument("--noise-smooth", type=int, default=2,
                   help="number of 3x3 smoothing passes; larger -> bigger eddies "
                        "(default: 2)")
    p.add_argument("--seed", type=int, default=0,
                   help="base RNG seed for reproducible noise (default: 0)")
    p.add_argument("--nlim", type=int, default=None,
                   help="hard cap on number of cycles (overrides YAML; handy for "
                        "smoke tests)")
    run(p.parse_args())
