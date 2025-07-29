#!/bin/bash
#SBATCH --job-name=cascaded_pure_swinir_sr
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH --time=48:00:00

echo "============================================"
echo "CASCADED PURE SWINIR TEMPERATURE SR (4x and 8x)"
echo "Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory allocated: 64GB per GPU"
echo "============================================"

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export APPTAINER_QUIET=1
export SINGULARITY_QUIET=1

# Change to project directory
cd $HOME/2x_swinir_temperature_sr_project

# Step 1: Verify environment
echo "============================================"
echo "Verifying environment..."
echo "============================================"

apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import torch
import numpy as np
import cv2
print('PyTorch:', torch.__version__)
print('NumPy:', np.__version__)
print('OpenCV:', cv2.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU Memory:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB')
try:
    import basicsr
    print('BasicSR:', basicsr.__version__)
except:
    print('BasicSR: Not found')
"

# Step 2: Install required packages if needed
echo "============================================"
echo "Checking/Installing required packages..."
echo "============================================"

apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user basicsr opencv-python timm matplotlib tqdm pillow

# Step 3: Create output directories
echo "============================================"
echo "Creating output directories..."
echo "============================================"

mkdir -p ./cascaded_swinir_results
mkdir -p ./cascaded_swinir_results/results_4x
mkdir -p ./cascaded_swinir_results/results_8x

# Step 4: Check for model file
echo "============================================"
echo "Checking for trained Pure SwinIR model..."
echo "============================================"

MODEL_PATH=""
# Try different possible locations
if [ -f "./experiments/PureTemperatureSR_SwinIR_x2_2nd_generation/models/net_g_64206.pth" ]; then
    MODEL_PATH="./experiments/PureTemperatureSR_SwinIR_x2_2nd_generation/models/net_g_64206.pth"
elif [ -f "./experiments/PureTemperatureSR_SwinIR_x2_2nd_generation/models/net_g_latest.pth" ]; then
    MODEL_PATH="./experiments/PureTemperatureSR_SwinIR_x2_2nd_generation/models/net_g_latest.pth"
else
    echo "Looking for available models..."
    find ./experiments -name "*.pth" -type f | head -10

    # Try to find any model
    FOUND_MODEL=$(find ./experiments -name "net_g_*.pth" -type f | head -1)
    if [ -n "$FOUND_MODEL" ]; then
        MODEL_PATH="$FOUND_MODEL"
        echo "Found model: $MODEL_PATH"
    else
        echo "ERROR: No trained model found!"
        echo "Please train a model first or specify the correct path"
        exit 1
    fi
fi

echo "Using model: $MODEL_PATH"
ls -lh $MODEL_PATH

# Step 5: Save the cascaded script (if not already present)
if [ ! -f "cascaded_swinir_temperature_sr.py" ]; then
    echo "============================================"
    echo "ERROR: cascaded_swinir_temperature_sr.py not found!"
    echo "Please ensure the script is in the project directory"
    echo "============================================"
    exit 1
fi

# Step 6: Run cascaded super-resolution
echo "============================================"
echo "Starting cascaded Pure SwinIR temperature SR..."
echo "Processing 5 samples with:"
echo "  - 4x: Model 2x → Model 2x"
echo "  - 8x: Model 2x → Model 2x → Model 2x"
echo "============================================"

# Use the same data directory as the cascaded_temperature_sr_unified.py
DATA_DIR="/home/vdidur/temperature_sr_project/data"

apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $HOME/2x_swinir_temperature_sr_project:$HOME/2x_swinir_temperature_sr_project \
    --bind $DATA_DIR:$DATA_DIR:ro \
    --env PYTHONPATH=$HOME/2x_swinir_temperature_sr_project:$PYTHONPATH \
    --env OMP_NUM_THREADS=4 \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python cascaded_swinir_temperature_sr.py \
    --npz-dir $DATA_DIR \
    --model-path $MODEL_PATH \
    --num-samples 5 \
    --save-dir ./cascaded_swinir_results \
    --overlap-ratio 0.75

# Check if processing was successful
if [ $? -eq 0 ]; then
    echo "============================================"
    echo "Cascaded processing completed successfully!"
    echo "============================================"

    # Show results summary
    echo ""
    echo "Results summary:"
    echo "----------------"
    echo "4x results:"
    ls -la ./cascaded_swinir_results/results_4x/ | head -5

    echo ""
    echo "8x results:"
    ls -la ./cascaded_swinir_results/results_8x/ | head -5

    echo ""
    echo "Comparison visualizations:"
    ls -la ./cascaded_swinir_results/*.png

    echo ""
    echo "Statistics report:"
    if [ -f "./cascaded_swinir_results/statistics_report.txt" ]; then
        cat ./cascaded_swinir_results/statistics_report.txt
    fi

else
    echo "============================================"
    echo "ERROR: Cascaded processing failed!"
    echo "Check the error log above for details"
    echo "============================================"
    exit 1
fi

echo ""
echo "============================================"
echo "Cascaded Pure SwinIR Temperature SR Finished: $(date)"
echo "============================================"

# Create archive of results
echo ""
echo "Creating compressed archive of results..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
tar -czf cascaded_swinir_results_${TIMESTAMP}.tar.gz ./cascaded_swinir_results/
echo "Archive created: cascaded_swinir_results_${TIMESTAMP}.tar.gz"
ls -lh cascaded_swinir_results_${TIMESTAMP}.tar.gz

# Show final GPU status
echo ""
echo "Final GPU status:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"