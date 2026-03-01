#!/bin/bash
#SBATCH -p raise
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 2:00:00
#SBATCH --job-name=openstl-full

echo "Job started at: $(date)"
echo "Node: $(hostname)"

source ~/.bashrc
conda activate openstl 2>/dev/null || true

pip install -e . -q 2>/dev/null
pip install pytest-cov -q 2>/dev/null

echo ""
echo "=== P0 + P1 Full Test Suite ==="
echo "Python: $(python --version)"
echo "Lightning: $(python -c 'import lightning; print(lightning.__version__)')"
echo "timm: $(python -c 'import timm; print(timm.__version__)')"
echo "torch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

cd /mnt/petrelfs/tancheng/work_dir/tancheng/OpenSTL

echo "=== Running All Tests ==="
pytest tests/test_imports.py \
       tests/test_methods/test_registration.py \
       tests/test_models/test_instantiation.py \
       tests/test_datasets/test_dataloaders.py \
       tests/test_core/test_metrics.py \
       tests/test_core/test_optim_scheduler.py \
       --cov=openstl --cov-report=term-missing -v --tb=short 2>&1

echo ""
echo "Job finished at: $(date)"
