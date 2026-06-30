#!/usr/bin/env python3
r"""
Jupiter global shallow-water model on the cubed sphere -- dry or moist --
distributed one face per GPU (6 ranks) via UCX.

  torchrun --nproc_per_node=6 jupiter_run.py --model dry   --out /data/tmp/jupiter_dry
  torchrun --nproc_per_node=6 jupiter_run.py --model moist --out /data/tmp/jupiter_moist

Each of the six gnomonic-equiangle cube faces lives on its own H100 (single
block per process -> no worker pool -> fast); snapy's cubed-sphere layout does
the cross-rank panel halo exchange over UCX (paddle.start_dist).

Physics:
  * Jupiter parameters (R, Omega, g) come from jupiter.yaml / CLI.
  * dry  : resting free surface + band-limited random noise; geostrophic
           adjustment spins up shallow-water turbulence and zonal jets.
  * moist: adds a vapour mixing ratio q (snapy scalar, advected) and the
           one-layer moisture closure of main.tex (eqs. 101-118) as an
           operator-split source each step -- precipitation W1, abyssal
           feedback W0, evaporation E, and the GLOBAL abyssal reservoir alpha0
           (its area mean is reduced across all six faces every step).
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as Fn
import paddle
import torch.distributed as dist
from scipy.ndimage import gaussian_filter
from snapy import Mesh, MeshOptions

from swe_noise import ab_to_lonlat
import paddle.cubed_sphere_remap as csr

# Jupiter constants
G = 24.79            # gravity (m/s^2); mean depth H0 set via --H0 (default 5000 m)
OMEGA = 1.7585e-4    # rotation rate (rad/s)
A = 71492000.0       # radius (m)
FACE_NAMES = ("+X", "+Y", "-X", "+Z", "-Y", "-Z")   # snapy CS face order


def turbulence_ic(face, x2v, x3v, rng, target_wind, sigma_cells, H0):
    """Jupiter-level balanced geostrophic-turbulence IC on one cube face.

    A smoothed random streamfunction psi gives a non-divergent wind field
    (contravariant vel2=-d psi/d x3, vel3=+d psi/d x2; on this near-uniform
    gnomonic grid contravariant ~ physical m/s). Winds are scaled to RMS
    `target_wind`, and the height is set in approximate geostrophic balance
    h' = (f a / g) psi so the start sheds little gravity-wave noise.

    Returns gh (=G*H), vel2, vel3 arrays of shape (nc3, nc2).
    """
    alpha, beta = np.meshgrid(x2v, x3v)              # (nc3, nc2): [k=x3, j=x2]
    lon, lat = ab_to_lonlat(FACE_NAMES[face], alpha, beta)

    psi = gaussian_filter(rng.standard_normal(alpha.shape), sigma_cells, mode="wrap")
    dx2 = x2v[1] - x2v[0]; dx3 = x3v[1] - x3v[0]
    vel2 = -np.gradient(psi, dx3, axis=0)            # -d psi / d x3
    vel3 = np.gradient(psi, dx2, axis=1)             # +d psi / d x2
    # tame random-gradient outliers (keep peak winds ~3x RMS for CFL stability)
    cap = 3.0 * np.sqrt((vel2**2 + vel3**2).mean())
    np.clip(vel2, -cap, cap, out=vel2); np.clip(vel3, -cap, cap, out=vel3)

    # rotate contravariant -> geographic to measure physical wind, then scale
    zero = np.zeros_like(vel2)
    gx, gy, gz = csr._local_contra_to_global_xyz(face, zero, vel2, vel3, alpha, beta)
    east = -np.sin(lon) * gx + np.cos(lon) * gy
    north = (-np.sin(lat) * np.cos(lon) * gx
             - np.sin(lat) * np.sin(lon) * gy + np.cos(lat) * gz)
    rms = float(np.sqrt((east**2 + north**2).mean()))
    s = target_wind / max(rms, 1e-12)
    vel2 *= s; vel3 *= s; psi *= s

    f = 2.0 * OMEGA * np.sin(lat)                     # Coriolis parameter
    hprime = (f * A / G) * psi                        # geostrophic height anomaly
    hprime = np.clip(hprime, -0.45 * H0, 0.45 * H0)   # keep the free surface positive
    gh = G * np.maximum(H0 + hprime, 0.1 * H0)
    return gh, vel2, vel3


def _uv_to_contra(face, alpha, beta, lon, lat, U, V):
    """(east U, north V) -> snapy contravariant (vel2, vel3) per cell (W92 method)."""
    def east_north(v2, v3):
        gx, gy, gz = csr._local_contra_to_global_xyz(
            face, np.zeros_like(v2), v2, v3, alpha, beta)
        east = -np.sin(lon) * gx + np.cos(lon) * gy
        north = (-np.sin(lat) * np.cos(lon) * gx
                 - np.sin(lat) * np.sin(lon) * gy + np.cos(lat) * gz)
        return east, north
    one = np.ones_like(alpha); zero = np.zeros_like(alpha)
    e1, n1 = east_north(one, zero)
    e2, n2 = east_north(zero, one)
    det = e1 * n2 - e2 * n1
    return (n2 * U - e2 * V) / det, (-n1 * U + e1 * V) / det


def global_forcing(face, alpha, beta, lon, lat, gseed, sigma_deg, nlon=288, nlat=145):
    """Globally-continuous non-divergent stirring sampled onto one cube face.

    A random streamfunction is built on a lon-lat grid (identical on every rank
    for a given gseed, so the field is continuous across panel seams), turned
    into a non-divergent (east,north) wind = k x grad psi, normalised to unit
    global RMS, sampled at this face's cells and rotated to contravariant.
    Returns (vel2, vel3) of shape lon.shape.
    """
    from scipy.interpolate import RegularGridInterpolator
    rng = np.random.default_rng(gseed)
    glon = np.linspace(0.0, 2 * np.pi, nlon, endpoint=False)
    glat = np.linspace(-np.pi / 2, np.pi / 2, nlat)
    sig = sigma_deg * nlon / 360.0
    psi = gaussian_filter(rng.standard_normal((nlat, nlon)), (sig, sig), mode=("nearest", "wrap"))
    dlat = glat[1] - glat[0]; dlon = glon[1] - glon[0]
    coslat = np.clip(np.cos(glat), 0.1, None)[:, None]
    ue = -np.gradient(psi, dlat, axis=0) / A
    un = np.gradient(psi, dlon, axis=1) / (A * coslat)
    rms = float(np.sqrt((ue**2 + un**2).mean())) or 1.0
    ue /= rms; un /= rms
    # periodic wrap in lon for interpolation
    glon_e = np.concatenate([glon, [2 * np.pi]])
    ue_e = np.concatenate([ue, ue[:, :1]], axis=1); un_e = np.concatenate([un, un[:, :1]], axis=1)
    pts = np.stack([lat.ravel(), (lon % (2 * np.pi)).ravel()], axis=1)
    U = RegularGridInterpolator((glat, glon_e), ue_e, bounds_error=False, fill_value=None)(pts).reshape(lon.shape)
    V = RegularGridInterpolator((glat, glon_e), un_e, bounds_error=False, fill_value=None)(pts).reshape(lon.shape)
    return _uv_to_contra(face, alpha, beta, lon, lat, U, V)


def global_scalar(lon, lat, gseed, sigma_deg, nlon=288, nlat=145):
    """Globally-continuous zero-mean unit-std random scalar field sampled onto a
    cube face (for seam-free HEIGHT forcing -- scalars exchange across panels
    with no basis rotation, unlike contravariant velocity components)."""
    from scipy.interpolate import RegularGridInterpolator
    rng = np.random.default_rng(gseed)
    glon = np.linspace(0.0, 2 * np.pi, nlon, endpoint=False)
    glat = np.linspace(-np.pi / 2, np.pi / 2, nlat)
    sig = sigma_deg * nlon / 360.0
    f = gaussian_filter(rng.standard_normal((nlat, nlon)), (sig, sig), mode=("nearest", "wrap"))
    f -= f.mean(); f /= (f.std() or 1.0)
    glon_e = np.concatenate([glon, [2 * np.pi]]); f_e = np.concatenate([f, f[:, :1]], axis=1)
    pts = np.stack([lat.ravel(), (lon % (2 * np.pi)).ravel()], axis=1)
    return RegularGridInterpolator((glat, glon_e), f_e, bounds_error=False, fill_value=None)(pts).reshape(lon.shape)


_BLUR_K = None


def _blur(x):
    """3x3 box smooth of a (1, nc3, nc2, 1) field (replicate-padded)."""
    global _BLUR_K
    if _BLUR_K is None:
        _BLUR_K = torch.ones(1, 1, 3, 3, device=x.device, dtype=x.dtype) / 9.0
    y = x[..., 0].unsqueeze(1)
    y = Fn.conv2d(Fn.pad(y, (1, 1, 1, 1), mode="replicate"), _BLUR_K)
    return y.squeeze(1).unsqueeze(-1)


def convective_source(v, p, dt, g3):
    """Radiative-convective moisture forcing (operator-split, in place).

    The flow is driven *spontaneously* by moist convection (no energetic IC):
      * radiative cooling  : Newtonian relaxation of gh toward gh_rad (uniform,
                             self-regulating -> the layer can't collapse);
      * evaporation        : surface flux moistening q toward q_surf;
      * convection / precip: where q>qs, P1 condenses vapour and removes mass
                             via latent heating (W1 = beta1 P1) -> local mass
                             deficit -> convergence -> vorticity -> turbulence.
    In statistical steady state radiative cooling ~ latent heating and
    evaporation ~ precipitation (radiative-convective equilibrium).

    Returns (mean precip rate, convecting area fraction) for diagnostics.
    """
    interior = lambda x: x[:, g3:-g3, g3:-g3, :]
    gh = interior(v["hydro_u"])[0]
    H = gh / G
    q = interior(v["scalar_r"])[0]

    gh = gh - (gh - p.gh_rad) / p.tau_rad * dt              # radiative cooling
    # Betts-Miller-style trigger: once q>qs, convection fires and consumes
    # moisture down toward q_ref (< qs) -> the cell overshoots to sub-saturation
    # and switches off, giving intermittent / localized (not uniform) convection
    trig = (q > p.qs).double()
    P1 = torch.clamp(H * (q - p.q_ref) / p.tau_c, min=0.0) * trig   # precipitation (m/s)
    dgh_lat = torch.clamp(G * p.beta1 * P1 * dt, max=p.maxdrop * gh)  # latent-heat mass sink
    gh_new = torch.clamp(gh - dgh_lat, min=G * p.hfloor)

    E = torch.clamp(H * (p.q_surf - q) / p.tau_evap, min=0.0)   # evaporation (m/s)
    s_new = torch.clamp(interior(v["scalar_s"])[0] + G * (E - P1) * dt, min=0.0)
    s_new = torch.minimum(s_new, p.qcap * gh_new)

    interior(v["hydro_u"])[0] = gh_new
    interior(v["hydro_w"])[0] = gh_new
    interior(v["scalar_s"])[0] = s_new
    interior(v["scalar_r"])[0] = torch.clamp(s_new / gh_new, min=0.0, max=p.qcap)
    torch.minimum(v["scalar_s"].clamp_(min=0.0),
                  p.qcap * v["hydro_u"][0].unsqueeze(0), out=v["scalar_s"])
    if p.scalar_diff > 0:
        v["scalar_s"] = (1 - p.scalar_diff) * v["scalar_s"] + p.scalar_diff * _blur(v["scalar_s"])
    return float(P1.mean()), float((q > p.qs).double().mean())


def thermal_source(v, coord, p, dt, g3):
    """Active 1.5-layer (thermal/Ripa) moist physics with constant-flux top
    cooling (main.tex section 4), operator-split after the snapy advection step.

    snapy carries the bulk reduced-gravity SWE with reference buoyancy b0 (the
    prognostic gh = b0 H), plus two advected scalars r0=b (buoyancy) and r1=q.
    Each step we add, in place:
      * the thermal-wind force  -grad( 1/2 (b-b0) gh^2 / b0 )  on the momentum
        (this is the buoyancy-anomaly part of the Ripa pressure  -grad(1/2 b H^2));
      * radiative cooling  db = -Q_R dt   and latent heating  db = +Lambda P/H dt;
      * evaporation E and precipitation P (with qs depending on b);
      * the condensation mass sink  dgh = -b0 beta P dt.
    Returns (mean precip, convecting fraction).
    """
    b0 = p.b0
    interior = lambda x: x[:, g3:-g3, g3:-g3, :]

    # ---- thermal-wind force from the advected buoyancy (full array) ----
    gh = v["hydro_u"][0]                       # (nc3f, nc2f, 1)
    b = v["scalar_r"][0]
    Pi = 0.5 * (b - b0) * gh * gh / b0          # buoyancy-anomaly pressure
    w2 = coord.center_width2(); w3 = coord.center_width3()   # PHYSICAL cell widths (m)
    g2 = torch.zeros_like(Pi); g3f = torch.zeros_like(Pi)
    g2[:, 1:-1, :] = (Pi[:, 2:, :] - Pi[:, :-2, :]) / (2 * w2[:, 1:-1, :])   # d/dx2 (physical)
    g3f[1:-1, :, :] = (Pi[2:, :, :] - Pi[:-2, :, :]) / (2 * w3[1:-1, :, :])  # d/dx3 (physical)
    interior(v["hydro_u"]).narrow(0, 2, 1)[0] -= interior(g2[None])[0] * dt  # vel2 momentum
    interior(v["hydro_u"]).narrow(0, 3, 1)[0] -= interior(g3f[None])[0] * dt  # vel3 momentum

    # ---- thermodynamics (interior) ----
    ghi = interior(v["hydro_u"])[0]
    H = ghi / b0
    bi = interior(v["scalar_r"])[0]
    q = interior(v["scalar_r"])[1]
    qs = p.qs0 * (1.0 + p.cqs * (bi - b0) / b0)          # Clausius-Clapeyron (linearised)
    # Betts-Miller trigger: where q>qs, convection consumes moisture toward
    # q_ref < qs so the cell overshoots sub-saturated and shuts off -> patchy
    # convection -> buoyancy gradients -> thermal-wind driving
    P1 = torch.clamp(H * (q - p.q_ref) / p.tau_c, min=0.0) * (q > qs).double()
    E = torch.clamp(H * (p.q_surf - q) / p.tau_e, min=0.0)

    b_new = bi + (-p.QR + p.Lambda * P1 / torch.clamp(H, min=p.hfloor)) * dt   # cooling + latent
    Hq_new = torch.clamp(H * q + (E - P1) * dt, min=0.0)
    gh_new = torch.clamp(ghi - b0 * p.beta * P1 * dt, min=b0 * p.hfloor)       # mass sink
    Hn = gh_new / b0
    q_new = torch.clamp(Hq_new / torch.clamp(Hn, min=p.hfloor), 0.0, p.qcap)

    interior(v["hydro_u"])[0] = gh_new
    interior(v["hydro_w"])[0] = gh_new
    interior(v["scalar_r"])[0] = b_new
    interior(v["scalar_r"])[1] = q_new
    interior(v["scalar_s"])[0] = gh_new * b_new
    interior(v["scalar_s"])[1] = gh_new * q_new
    # light diffusion of both scalars (stability of the advected fields)
    if p.scalar_diff > 0:
        for k in range(2):
            v["scalar_s"][k:k+1] = ((1 - p.scalar_diff) * v["scalar_s"][k:k+1]
                                    + p.scalar_diff * _blur(v["scalar_s"][k:k+1]))
    return float(P1.mean()), float((q > qs).double().mean())


def run(args):
    device = paddle.start_dist("ucx")          # sets cuda device + process group
    rank = dist.get_rank() if dist.is_initialized() else 0
    g3 = 3

    moist = args.model == "moist"
    opt = MeshOptions.from_yaml(args.config)
    opt.set_local_horizontal_cells(args.N, args.N)
    opt.blocks_per_process(1)
    if args.visc > 0:                              # metric-correct Laplacian viscosity
        from snapy import DiffusionOptions
        do = DiffusionOptions(); do.nu_iso(args.visc)
        opt.block().hydro().diffusion(do)
    if moist:
        opt.block().intg().cfl(args.cfl)        # lower CFL for stiff moist source
    opt.block().output_dir(args.out)
    opt.block().intg().tlim(args.days * 86400.0)
    if moist:
        sc = opt.block().scalar(); sc.nvar(2); sc.names(["buoy", "qv"]); opt.block().scalar(sc)
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
    dist.barrier()
    mesh = Mesh(opt); mesh.to(device)

    # ---- initial condition on this rank's single face ----
    block = mesh.blocks[0]
    face = block.get_layout().loc_of(rank)[0]              # which cube face this rank owns
    coord = block.module("coord")
    x2v = coord.buffer("x2v").cpu().numpy(); x3v = coord.buffer("x3v").cpu().numpy()
    nc2, nc3 = x2v.size, x3v.size
    rng = np.random.default_rng(args.seed + 1000 * rank)   # independent field per face
    w = torch.zeros((4, nc3, nc2, 1), dtype=torch.float64)
    if moist:
        # active 1.5-layer: REST, neutral buoyancy b=b0, sub-saturated q + seed;
        # constant-flux top cooling spins convection (hence the flow) up from rest
        args.b0 = G
        w[0, :, :, 0] = args.b0 * args.H0          # gh = b0 H
        var0 = {"hydro_w": w.to(device)}
        rb = np.full((nc3, nc2), args.b0)
        rq = args.q_init + args.q_seed * rng.standard_normal((nc3, nc2))
        r = np.stack([rb, rq])                      # (2, nc3, nc2)
        var0["scalar_r"] = torch.from_numpy(r).reshape(2, nc3, nc2, 1).to(device)
    elif args.force_amp > 0:
        # dry forced-dissipative: start from REST; continuous stochastic stirring
        # + linear drag spin the flow up to a statistically-steady jet state
        w[0, :, :, 0] = G * args.H0
        var0 = {"hydro_w": w.to(device)}
    else:
        # dry control: balanced random-streamfunction turbulence at target_wind
        gh, vel2, vel3 = turbulence_ic(face, x2v, x3v, rng,
                                       args.target_wind, args.eddy_cells, args.H0)
        w[0, :, :, 0] = torch.from_numpy(gh)
        w[2, :, :, 0] = torch.from_numpy(vel2)
        w[3, :, :, 0] = torch.from_numpy(vel3)
        var0 = {"hydro_w": w.to(device)}
    block_vars, current_time = mesh.initialize([var0])
    forced = (not moist) and args.force_amp > 0
    F2 = F3 = None
    refresh = max(1, int(args.force_tau / 200.0))   # regenerate stirring every ~force_tau
    if forced:                                       # face geometry for global forcing
        falpha, fbeta = np.meshgrid(x2v, x3v)
        flon, flat = ab_to_lonlat(FACE_NAMES[face], falpha, fbeta)

    # ---- time integration ----
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

        pbar = cfrac = 0.0
        if moist:
            pbar, cfrac = thermal_source(block_vars[0], coord, args, dt, g3)
        elif forced:
            # refresh the global (seam-free) random HEIGHT pattern every ~force_tau;
            # gseed depends only on the refresh index -> identical on all ranks
            if F2 is None or cycle % refresh == 0:
                gseed = args.seed * 100003 + (cycle // refresh)
                fp = global_scalar(flon, flat, gseed, args.force_scale_deg)
                F2 = torch.from_numpy(fp).reshape(1, nc3, nc2, 1).to(device)
            v = block_vars[0]
            interior = lambda x: x[:, g3:-g3, g3:-g3, :]
            drag = dt / args.drag_tau
            # scalar height forcing (seam-free) -> geostrophic adjustment -> jets;
            # plus Rayleigh drag on momentum
            kick = args.force_amp * dt
            interior(v["hydro_u"]).narrow(0, 0, 1).add_(interior(F2) * kick)
            interior(v["hydro_w"]).narrow(0, 0, 1).add_(interior(F2) * kick)
            interior(v["hydro_u"]).narrow(0, 1, 3).mul_(1.0 - drag)
            interior(v["hydro_w"]).narrow(0, 1, 3).mul_(1.0 - drag)

        current_time += dt
        mesh.make_outputs(block_vars, current_time)
        if rank == 0 and cycle % args.diag_every == 0:
            msg = f"[cycle {cycle}] t={current_time/86400:.1f} d  dt={dt:.1f} s"
            if moist:
                msg += f"  precip~{pbar:.2e} m/s  convecting={cfrac*100:.1f}%"
            print(msg, flush=True)
    mesh.finalize(block_vars, current_time)
    paddle.close_dist()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("dry", "moist"), required=True)
    p.add_argument("-c", "--config",
                   default=os.path.join(os.path.dirname(__file__), "jupiter.yaml"))
    p.add_argument("--out", required=True, help="output dir (e.g. /data/tmp/jupiter_dry)")
    p.add_argument("--N", type=int, default=768, help="cells per face (default 768 ~100 km)")
    p.add_argument("--days", type=float, default=1000.0, help="simulated days (default 1000)")
    p.add_argument("--H0", type=float, default=5000.0, help="mean weather-layer depth, m")
    # turbulence IC
    p.add_argument("--target-wind", type=float, default=80.0, dest="target_wind",
                   help="RMS wind speed of the injected turbulence, m/s (default 80)")
    p.add_argument("--eddy-cells", type=float, default=16.0, dest="eddy_cells",
                   help="stirring/streamfunction smoothing sigma in cells (eddy scale)")
    p.add_argument("--seed", type=int, default=0)
    # --- dry forced-dissipative turbulence (continuous stirring + drag) ---
    p.add_argument("--force-amp", type=float, default=0.0, dest="force_amp",
                   help="height (geopotential) forcing rate, m^2/s^3 (>0 enables forced dry run from rest; seam-free)")
    p.add_argument("--force-tau", type=float, default=2.0e4, dest="force_tau",
                   help="stirring decorrelation time, s (pattern refreshed each interval)")
    p.add_argument("--force-scale-deg", type=float, default=12.0, dest="force_scale_deg",
                   help="forcing length scale in degrees (global streamfunction smoothing)")
    p.add_argument("--drag-tau", type=float, default=8.64e5, dest="drag_tau",
                   help="linear (Rayleigh) drag time, s (~10 days; sets equilibrium wind)")
    p.add_argument("--visc", type=float, default=0.0,
                   help="isotropic Laplacian viscosity, m^2/s (damps grid/seam noise)")
    # --- active 1.5-layer (thermal) moist forcing: convection drives the flow ---
    p.add_argument("--QR", type=float, default=1.0e-7,
                   help="constant top radiative cooling rate of buoyancy, m/s^3")
    p.add_argument("--Lambda", type=float, default=8.0,
                   help="latent-buoyancy factor Lg/(Cp theta0), m/s^2")
    p.add_argument("--beta", type=float, default=1.0, help="condensation mass-sink factor")
    p.add_argument("--qs0", type=float, default=0.02, help="reference saturation mixing ratio at b0")
    p.add_argument("--cqs", type=float, default=3.0,
                   help="Clausius-Clapeyron sensitivity: qs = qs0(1 + cqs (b-b0)/b0)")
    p.add_argument("--q-surf", type=float, default=0.03, dest="q_surf",
                   help="surface value evaporation moistens q toward")
    p.add_argument("--q-init", type=float, default=0.018, dest="q_init",
                   help="initial (sub-saturated) mixing ratio")
    p.add_argument("--q-ref", type=float, default=0.016, dest="q_ref",
                   help="Betts-Miller: convection depletes q toward this (<qs0)")
    p.add_argument("--q-seed", type=float, default=1.0e-3, dest="q_seed",
                   help="amplitude of the initial random q perturbation")
    p.add_argument("--tau-e", type=float, default=3.0e5, dest="tau_e",
                   help="evaporation time, s (~3.5 days)")
    p.add_argument("--tau-c", type=float, default=2.0e4, dest="tau_c",
                   help="convective/precip time, s (~0.2 day)")
    p.add_argument("--scalar-diff", type=float, default=0.04, dest="scalar_diff",
                   help="per-step scalar diffusion fraction (stability of b, q)")
    p.add_argument("--cfl", type=float, default=0.5, help="CFL for the moist run")
    p.add_argument("--hfloor", type=float, default=100.0)
    p.add_argument("--qcap", type=float, default=0.10)
    p.add_argument("--diag-every", type=int, default=2000)
    run(p.parse_args())
