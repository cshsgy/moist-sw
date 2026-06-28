# moist-sw — global shallow water on the cubed sphere with random-noise initial conditions

A small, self-contained global **shallow-water model** built on
[**snapy**](https://github.com/chengcli/snapy), run on the gnomonic-equiangle
**cubed sphere** (all six faces). Instead of a balanced analytic initial state
like the bundled Williamson-1992 (**W92**) benchmarks, this case starts from a
**flat ocean at rest plus random noise of a configurable scale** injected into
the free-surface height. The Coriolis force then drives geostrophic adjustment,
radiating gravity waves and spinning up freely-evolving shallow-water
turbulence.

It reuses snapy's cubed-sphere shallow-water machinery — the `shallow-water`
equation of state, the `shallow-roe` Riemann solver, and the `coriolis`
forcing — the same components the W92 example uses, but with a trivially
specified (resting) background so **no `paddle` dependency is required**: it runs
on `snapy` alone.

| file | purpose |
|------|---------|
| `swe_noise.py`   | driver: builds the cubed-sphere mesh, injects the noisy IC, time-steps |
| `swe_noise.yaml` | configuration: geometry, cubed-sphere layout, solver, Coriolis, output |
| `plot_height.py` | plot the free-surface height from the NetCDF output |
| `requirements.txt` | Python dependencies (`snapy` + plotting) |

![example free-surface height at t=0 (noise injected)](docs/example_height.png)

## Installation

Requires Python 3.9–3.13 on Linux (x86_64) or macOS (ARM64). Use a virtual
environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # installs snapy, matplotlib, netCDF4, ...
```

`pip install snapy` pulls in `torch`, `numpy`, `netCDF4`, and the rest of the
snapy stack automatically.

## Running

The driver is launched with `torchrun` (snapy's distributed entry point). A
single process holds all six cube faces (`blocks_per_process: 6`):

```bash
# CPU, one process, all six faces
torchrun --nproc_per_node=1 swe_noise.py --device cpu --output-dir out_noise
```

NetCDF files (`swe_noise.out0.*.nc`) are written to `--output-dir` every 6
simulated hours (see `outputs.dt` in the YAML), for `tlim` = 15 days.

A quick smoke test (a few cycles, finishes in seconds):

```bash
torchrun --nproc_per_node=1 swe_noise.py --device cpu --output-dir out_smoke --nlim 5
```

### Tuning the injected noise

Two CLI flags set "the scale" of the random perturbation added to the resting
height field (`H0 = 8000 m`):

| flag | meaning | default |
|------|---------|---------|
| `--noise-amp`    | standard deviation of the height perturbation, in **metres** (amplitude scale) | `50` |
| `--noise-smooth` | number of 3×3 smoothing passes; larger ⇒ longer correlation length / bigger eddies (length scale) | `2` |
| `--seed`         | base RNG seed; the run is fully reproducible for a fixed seed | `0` |

```bash
# larger-amplitude, larger-scale blobs, reproducible
torchrun --nproc_per_node=1 swe_noise.py --device cpu \
    --output-dir out_big --noise-amp 120 --noise-smooth 6 --seed 42
```

### Multi-GPU

Set the backend to `nccl` in `swe_noise.yaml` (`distribute.backend`) and launch
one rank per face:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun --nproc_per_node=6 swe_noise.py --device cuda --output-dir out_noise
```

The driver only initializes a `torch.distributed` process group when
`WORLD_SIZE > 1`, so the single-process CPU command above needs no extra setup.

## Plotting

```bash
python plot_height.py out_noise/swe_noise.out0.00000.nc -o height.png
# -t <index> selects a time slice (default: last)
```

snapy unrolls the six faces into one array with 2D `lon`/`lat` coordinate
variables; the script scatters every cell at its (lon, lat) and colours it by
height `h = rho / g` (the shallow-water prognostic `rho` is the geopotential
`gh`).

## What to expect

At `t = 0` the height map is the injected noise (band-limited blobs around
8000 m, with the cubed-sphere panel structure faintly visible near the poles).
As the run proceeds, the unbalanced height field adjusts geostrophically:
gravity waves radiate, vortices form and merge, and the flow organizes into the
banded, eddy-rich structure characteristic of rotating shallow-water turbulence.

## Notes

- The noise is generated independently per face from `--seed` (seed + 1000·face)
  and smoothed within each face; cross-face continuity is handled by the
  solver's own panel exchange once the run starts, so small seams at panel
  edges in the very first snapshot are expected and quickly smooth out.
- Configuration knobs (resolution `nx2`/`nx3`, planet radius, rotation rate
  `omega1`, CFL, `tlim`, output cadence) live in `swe_noise.yaml`.
