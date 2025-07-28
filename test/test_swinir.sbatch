#!/bin/bash
#SBATCH --job-name=test_swinir_sr_2x
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=02:00:00

echo "============================================"
echo "Testing Pure SwinIR SR Model: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set environment variables to suppress warnings
export APPTAINER_QUIET=1
export SINGULARITY_QUIET=1

# Project directory
PROJECT_DIR="$HOME/2x_swinir_temperature_sr_project"
cd $PROJECT_DIR

# Create test results directory
mkdir -p test_swinir_results

echo "============================================"
echo "Checking environment..."
echo "============================================"

# Install required packages if needed
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user basicsr opencv-python timm matplotlib tqdm

# Test environment
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import sys
print(f'Python: {sys.version}')
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ PyTorch: {e}')
try:
    import basicsr
    print(f'✅ BasicSR: {basicsr.__version__}')
except Exception as e:
    print(f'❌ BasicSR: {e}')
"

echo "============================================"
echo "Looking for trained models..."
echo "============================================"

# Find the latest model checkpoint
MODEL_DIR="$PROJECT_DIR/experiments/PureTemperatureSR_SwinIR_x2_2nd_generation/models"
if [ -d "$MODEL_DIR" ]; then
    echo "Checking model directory: $MODEL_DIR"
    ls -la $MODEL_DIR/net_g_*.pth 2>/dev/null || echo "No model checkpoints found yet"

    # Find the latest checkpoint
    LATEST_MODEL=$(ls -t $MODEL_DIR/net_g_*.pth 2>/dev/null | head -1)
    if [ -z "$LATEST_MODEL" ]; then
        echo "❌ No trained model found in $MODEL_DIR"
        echo "Please ensure training has completed and saved checkpoints"
        exit 1
    else
        echo "✅ Found model: $LATEST_MODEL"
    fi
else
    echo "❌ Model directory not found: $MODEL_DIR"
    echo "Please check if training has been run"
    exit 1
fi

echo "============================================"
echo "Testing Pure SwinIR Model..."
echo "============================================"

# Run the test script
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $PROJECT_DIR:$PROJECT_DIR \
    --bind /home/vdidur/temperature_sr_project/data:/home/vdidur/temperature_sr_project/data:ro \
    --env PYTHONPATH=$PROJECT_DIR:$PYTHONPATH \
    --workdir $PROJECT_DIR \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python test_swinir_temperature_sr.py

# Check results
if [ $? -eq 0 ]; then
    echo "============================================"
    echo "✅ Testing completed successfully!"
    echo "============================================"

    # Show test results summary
    echo ""
    echo "Test results have been displayed in the console output above."
    echo "Look for the PURE SWINIR TEST SUMMARY section for average metrics."

    # Check if visualization results were saved
    if [ -d "./test_swinir_results" ] && [ "$(ls -A ./test_swinir_results 2>/dev/null)" ]; then
        echo ""
        echo "Saved files in test_swinir_results/:"
        ls -la test_swinir_results/
    fi
else
    echo "============================================"
    echo "❌ Testing failed!"
    echo "Check the error messages above for details"
    echo "============================================"
fi

# Clean up GPU memory
echo ""
echo "Cleaning up GPU memory..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('✅ GPU memory cleared')
else:
    print('No GPU available for cleanup')
"

echo ""
echo "============================================"
echo "Pure SwinIR Testing Finished: $(date)"
echo "============================================"