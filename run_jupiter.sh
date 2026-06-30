#!/bin/bash
# Production Jupiter SW runs: dry then moist, 1000 days each, 6 GPUs (1 face/GPU).
VENV=/home/sam2/dev/moist-sw/.venv
cd /home/sam2/dev/moist-sw
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
echo "=== DRY start $(date -u) ==="
$VENV/bin/torchrun --nproc_per_node=6 jupiter_run.py --model dry \
    --out /data/tmp/jupiter_dry --days 1000 --H0 5000 --diag-every 2000 \
    > /data/tmp/jupiter_dry.log 2>&1
echo "=== DRY done $(date -u), exit=$? ==="
echo "=== MOIST start $(date -u) ==="
$VENV/bin/torchrun --nproc_per_node=6 jupiter_run.py --model moist \
    --out /data/tmp/jupiter_moist --days 1000 --H0 5000 --diag-every 2000 \
    > /data/tmp/jupiter_moist.log 2>&1
echo "=== MOIST done $(date -u), exit=$? ==="
