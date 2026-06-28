#!/usr/bin/env python3
r"""
Moist shallow-water model on the gnomonic-equiangle cubed sphere.

Implements "The one-layer model with moisture" of writing/main.tex
(eqs. 101-118): a single *weather layer* (height H, column-mean velocity v,
vapour mixing ratio q) coupled to an *abyssal* water reservoir alpha0. This is
the closed reduction of the two-layer (weather + infinitely-deep abyssal) moist
SW system in the same document.

Strategy (chosen: build on the snapy shallow-water core)
--------------------------------------------------------
The dry shallow-water dynamics are integrated by snapy exactly as in the dry
`swe_noise.py` case:

    snapy prognostic gh  = g' H            (geopotential of the weather layer)
                     gh v                   (momentum)
    snapy scalar     r   = q                (vapour mixing ratio, advected as gh*q)

so snapy supplies advection, Coriolis, and the geopotential pressure gradient.
The *moisture physics* are added as an operator-split source applied to the
conserved state every step (the same pattern the repo's HS94 / hot-Jupiter
forcings use):

    precipitation         P1 = max(H (q - qs) / tau1, 0)        (relaxation, eq. 108)
    top vertical velocity  W1 = beta1 P1                         (eq. 68, beta1 = L/(Cp dtheta))
    abyssal feedback       W0 = alpha0 / (tau2 q0)               (eq. 111)
    evaporation            E  = W0 q0                            (eq. 84)

    d_t H        =  W0 - W1                                       (eq. 102)
    d_t (H q)    =  W0 q0 - P1                                    (eq. 114)
    d_t alpha0   =  <P1>_area - alpha0 / tau2                     (eq. 117)

In snapy units (gh = g' H, conserved vapour s = gh q):

    d_t gh   = g' (W0 - W1)
    d_t s    = g' (W0 q0 - P1)

A balanced steady state needs beta1 = 1/q0 (the document's beta1 ~ 1/q1); that
is the default. Modelling notes:

  * eq. 105's momentum RHS as written omits the geopotential pressure-gradient
    term; we keep snapy's standard shallow-water pressure gradient (the physical
    closure) with reduced gravity g' = g theta1/theta0 (`--gprime`).
  * qs is taken as a constant parameter (`--qs`) rather than the full q_s(P,T).
  * alpha0's area integral uses a uniform (equal-weight) cell mean -- a mild
    approximation on the cubed sphere, adequate for a bulk reservoir.
  * a positivity limiter caps the per-step fractional mass loss so a column that
    precipitates faster than it is resupplied thins but never goes negative.

Run (single process, all six faces, CPU):

    torchrun --nproc_per_node=1 moist_sw.py --device cpu --output-dir out_moist

See README.md for details.
"""
import argparse
import os

import numpy as np
import torch
from snapy import Mesh, MeshOptions

# reuse the dry-case helpers: cube-face names, lon/lat map, noisy height field
import swe_noise as dry
from swe_noise import FACE_NAMES, noisy_height


def moisture_source(block_vars, alpha0, p, dt, gp, g3):
    """Apply one operator-split moisture step in place; return updated alpha0.

    block_vars : list of per-face dicts with hydro_u/hydro_w/scalar_s/scalar_r
    alpha0     : abyssal reservoir (m), a single global scalar
    p          : parameter namespace (qs, q0, tau1, tau2, beta1, ...)
    gp         : reduced gravity g'
    g3         : number of ghost cells (interior = [g3:-g3] in x2, x3)
    """
    interior = lambda x: x[:, g3:-g3, g3:-g3, :]
    W0 = alpha0 / (p.tau2 * p.q0)            # abyssal feedback velocity (m/s)

    p1_sum = 0.0
    n_cells = 0
    for v in block_vars:
        gh = interior(v["hydro_u"])[0]        # g' H  (geopotential), shape (nc3,nc2,1)
        H = gh / gp
        q = interior(v["scalar_r"])[0]        # vapour mixing ratio

        P1 = torch.clamp(H * (q - p.qs) / p.tau1, min=0.0)   # precip (m/s)
        W1 = p.beta1 * P1

        # continuity (eq. 102):  d_t gh = g' (W0 - W1); limit fractional mass
        # loss for positivity. The source acts on the MASS only -- eq. 105 has
        # no moisture source on the momentum H v, so the conserved momentum
        # hydro_u[1:4] is left to the dynamics and velocity = (H v)/H adjusts.
        dH = torch.clamp((W0 - W1) * dt, min=-p.maxdrop * H)
        gh_new = torch.clamp(gh + gp * dH, min=gp * p.hfloor)

        # vapour (eq. 114):  d_t (gh q) = g' (W0 q0 - P1)
        s_new = torch.clamp(interior(v["scalar_s"])[0] + gp * (W0 * p.q0 - P1) * dt, min=0.0)
        q_new = torch.clamp(s_new / gh_new, min=0.0, max=p.qcap)

        # write back (mass + vapour); momentum is untouched
        interior(v["hydro_u"])[0] = gh_new
        interior(v["hydro_w"])[0] = gh_new
        interior(v["scalar_s"])[0] = s_new
        interior(v["scalar_r"])[0] = q_new

        p1_sum += float(P1.sum())
        n_cells += P1.numel()

    # abyssal reservoir:  d_t alpha0 = <P1>_area - alpha0 / tau2
    p1_bar = p1_sum / max(n_cells, 1)
    alpha0 += (p1_bar - alpha0 / p.tau2) * dt
    return alpha0, p1_bar


def run(args):
    gp = args.gprime
    if args.beta1 is None:
        args.beta1 = 1.0 / args.q0        # balanced default (doc: beta1 ~ 1/q1)

    # ---- distributed only when launched with WORLD_SIZE > 1 ------------------
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        import torch.distributed as dist
        import torch.distributed.distributed_c10d as dist_c10d
        import snapy
        import yaml
        with open(args.config) as f:
            backend = yaml.safe_load(f).get("distribute", {}).get("backend", "gloo")
        dist.init_process_group(backend=backend, init_method="env://")
        snapy.distributed.set_process_group(dist_c10d._get_default_group())

    if args.device == "cuda" and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # ---- build the mesh (one vapour scalar) ----------------------------------
    opt = MeshOptions.from_yaml(args.config)
    opt.block().output_dir(args.output_dir)
    if args.nlim is not None:
        opt.block().intg().nlim(args.nlim)
    os.makedirs(args.output_dir, exist_ok=True)
    mesh = Mesh(opt)
    mesh.to(device)
    g3 = 3   # nghost from the YAML

    # ---- initial condition ---------------------------------------------------
    # weather layer: resting, mean depth H0 + band-limited noise; uniform vapour q_init
    block_vars = []
    for f, block in enumerate(mesh.blocks):
        coord = block.module("coord")
        x2v = coord.buffer("x2v").cpu().numpy()
        x3v = coord.buffer("x3v").cpu().numpy()
        nc2, nc3 = x2v.size, x3v.size
        alpha, beta = np.meshgrid(x2v, x3v)
        rng = np.random.default_rng(args.seed + 1000 * f)
        H = noisy_height(FACE_NAMES[f], alpha, beta, rng, args.noise_amp, args.noise_smooth)

        w = torch.zeros((4, nc3, nc2, 1), dtype=torch.float64)
        w[0, :, :, 0] = torch.from_numpy(gp * H)   # geopotential g' H
        r = torch.zeros((1, nc3, nc2, 1), dtype=torch.float64)
        r[0, :, :, 0] = args.q_init                 # vapour mixing ratio
        block_vars.append({"hydro_w": w.to(device), "scalar_r": r.to(device)})

    block_vars, current_time = mesh.initialize(block_vars)
    alpha0 = args.alpha0

    # ---- time integration with operator-split moisture source ----------------
    intg = mesh.module("block0.intg")
    cycle = 0
    mesh.make_outputs(block_vars, current_time)
    while not intg.stop(cycle, current_time):
        cycle += 1
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(block_vars)
        mesh.print_cycle_info(block_vars, current_time, dt)

        # dry shallow-water dynamics (advection + Coriolis + pressure gradient)
        for stage in range(len(intg.stages)):
            mesh.forward(block_vars, dt, stage)
        err = mesh.check_redo(block_vars)
        if err > 0:
            continue
        if err < 0:
            break

        # moisture physics (operator split)
        alpha0, p1_bar = moisture_source(block_vars, alpha0, args, dt, gp, g3)

        current_time += dt
        if cycle % args.diag_every == 0:
            print(f"  [moist] alpha0={alpha0:.3e} m  <P1>={p1_bar:.3e} m/s")
        mesh.make_outputs(block_vars, current_time)
    mesh.finalize(block_vars, current_time)

    if world_size > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config",
                   default=os.path.join(os.path.dirname(__file__), "moist_sw.yaml"))
    p.add_argument("--output-dir", default="out_moist")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    # --- moisture parameters (eqs. 101-118) ---
    p.add_argument("--gprime", type=float, default=dry.G,
                   help="reduced gravity g' = g*theta1/theta0 (default: g)")
    p.add_argument("--qs", type=float, default=0.02,
                   help="saturation mixing ratio (default: 0.02)")
    p.add_argument("--q0", type=float, default=0.05,
                   help="abyssal-layer mixing ratio (default: 0.05)")
    p.add_argument("--beta1", type=float, default=None,
                   help="latent-heating factor beta1 = L/(Cp*dtheta); "
                        "default 1/q0 (balanced steady state)")
    p.add_argument("--tau1", type=float, default=1.0e5,
                   help="precipitation relaxation time, s (default: 1e5)")
    p.add_argument("--tau2", type=float, default=5.0e4,
                   help="abyssal evaporation/resupply time, s (default: 5e4)")
    p.add_argument("--q-init", type=float, default=0.025, dest="q_init",
                   help="initial weather-layer mixing ratio (default: 0.025)")
    p.add_argument("--alpha0", type=float, default=0.0,
                   help="initial abyssal reservoir, m (default: 0)")
    # --- positivity / safety limiters ---
    p.add_argument("--hfloor", type=float, default=100.0,
                   help="minimum weather-layer depth, m (default: 100)")
    p.add_argument("--qcap", type=float, default=1.0,
                   help="cap on mixing ratio (default: 1.0)")
    p.add_argument("--maxdrop", type=float, default=0.2,
                   help="max fractional layer-depth loss per step (default: 0.2)")
    # --- initial-noise scale (reused from the dry case) ---
    p.add_argument("--noise-amp", type=float, default=50.0,
                   help="std-dev of the initial height perturbation, m (default: 50)")
    p.add_argument("--noise-smooth", type=int, default=2,
                   help="3x3 smoothing passes for the noise (default: 2)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    p.add_argument("--nlim", type=int, default=None,
                   help="hard cap on cycles (overrides YAML; handy for smoke tests)")
    p.add_argument("--diag-every", type=int, default=200, dest="diag_every",
                   help="print moisture diagnostics every N cycles (default: 200)")
    run(p.parse_args())
