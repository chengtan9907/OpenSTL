#!/bin/bash
#SBATCH -p raise
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:00:00
#SBATCH --job-name=openstl-test

echo "Job started at: $(date)"
echo "Node: $(hostname)"

# Load environment
source ~/.bashrc
conda activate openstl 2>/dev/null || true

# Install dependencies if needed
pip install -e . -q 2>/dev/null
pip install pytest-cov -q 2>/dev/null

echo ""
echo "=== Running OpenSTL Tests ==="
echo "Python: $(python --version)"
echo "Lightning: $(python -c 'import lightning; print(lightning.__version__)')"
echo "timm: $(python -c 'import timm; print(timm.__version__)')"
echo "torch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Run tests - only the new compatibility tests
cd /mnt/petrelfs/tancheng/work_dir/tancheng/OpenSTL
echo ""
echo "=== Running Import Tests ==="
pytest tests/test_imports.py -v --tb=short 2>&1

echo ""
echo "=== Running Method Registration Tests ==="
pytest tests/test_methods/test_registration.py -v --tb=short 2>&1

echo ""
echo "=== Running Model Instantiation Tests ==="
pytest tests/test_models/test_instantiation.py -v --tb=short 2>&1

echo ""
echo "=== Running Dataloader Tests ==="
pytest tests/test_datasets/test_dataloaders.py -v --tb=short 2>&1

echo ""
echo "=== Coverage Report ==="
pytest tests/test_imports.py tests/test_methods/test_registration.py tests/test_models/test_instantiation.py tests/test_datasets/test_dataloaders.py --cov=openstl --cov-report=term-missing 2>&1

echo ""
echo "Job finished at: $(date)"
