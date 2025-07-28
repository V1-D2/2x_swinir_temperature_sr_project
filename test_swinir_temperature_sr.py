#!/usr/bin/env python3
"""
Test script for Pure SwinIR Temperature Super-Resolution Model
Tests pretrained model on AMSR-2 temperature data
"""

import os
import sys
import torch
import numpy as np
import glob
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Import model components
from swinir_model import PureSwinIRModel
from data_preprocessing import TemperatureDataPreprocessor
from config_temperature import *
from utils import calculate_psnr, calculate_ssim
from basicsr.utils import tensor2img, imwrite

# Flag to control file saving
SAVE_FILES = False  # Set to True to save images and NPZ files

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_pure_swinir_model(model_path: str, device: torch.device) -> PureSwinIRModel:
    """Load pretrained Pure SwinIR Model"""

    # Create configuration from config_temperature.py
    opt = {
        'name': name,
        'model_type': model_type,
        'scale': scale,
        'num_gpu': 1,
        'network_g': network_g,
        # No network_d for pure SwinIR
        'path': path,
        'train': train,
        'is_train': False,
        'dist': False
    }

    # Create model
    model = PureSwinIRModel(opt)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Determine checkpoint format and load weights
    if isinstance(checkpoint, dict) and 'params' in checkpoint:
        model.net_g.load_state_dict(checkpoint['params'], strict=True)
        logger.info(f"Loaded model from iteration {checkpoint.get('iter', 'unknown')}")
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.net_g.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        model.net_g.load_state_dict(checkpoint, strict=True)

    model.net_g.eval()
    model.net_g.to(device)

    logger.info(f"✓ Pure SwinIR model loaded from {model_path}")
    return model


def test_single_sample(model: PureSwinIRModel,
                       temperature: np.ndarray,
                       preprocessor: TemperatureDataPreprocessor,
                       device: torch.device) -> Dict:
    """Test model on single temperature sample"""

    # Preprocess temperature data
    temperature = preprocessor.crop_or_pad(temperature)

    # Save original min/max for denormalization
    temp_min = np.min(temperature)
    temp_max = np.max(temperature)

    # Normalize temperature
    temperature_norm = preprocessor.normalize_temperature(temperature)

    # Create LR-HR pair
    lr, hr = preprocessor.create_lr_hr_pair(temperature_norm, scale_factor=2)

    # Convert to tensors
    lr_tensor = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).float().to(device)
    hr_tensor = torch.from_numpy(hr).unsqueeze(0).unsqueeze(0).float().to(device)

    # Run model inference
    with torch.no_grad():
        start_time = time.time()
        sr_tensor = model.net_g(lr_tensor)
        sr_tensor = torch.clamp(sr_tensor, 0, 1)
        inference_time = time.time() - start_time

    # Convert back to numpy
    lr_np = lr_tensor.cpu().numpy()[0, 0]
    hr_np = hr_tensor.cpu().numpy()[0, 0]
    sr_np = sr_tensor.cpu().numpy()[0, 0]

    # Denormalize to get temperature values
    lr_temp = lr_np * (temp_max - temp_min) + temp_min
    hr_temp = hr_np * (temp_max - temp_min) + temp_min
    sr_temp = sr_np * (temp_max - temp_min) + temp_min

    # Calculate metrics using BasicSR tensor2img (expects [0,1] range)
    lr_img = tensor2img([lr_tensor])
    hr_img = tensor2img([hr_tensor])
    sr_img = tensor2img([sr_tensor])

    psnr = calculate_psnr(sr_img, hr_img, crop_border=0, test_y_channel=False)
    ssim = calculate_ssim(sr_img, hr_img, crop_border=0, test_y_channel=False)

    # Calculate temperature errors
    temp_error_mean = np.mean(np.abs(sr_temp - hr_temp))
    temp_error_max = np.max(np.abs(sr_temp - hr_temp))

    return {
        'lr_temp': lr_temp,
        'hr_temp': hr_temp,
        'sr_temp': sr_temp,
        'lr_norm': lr_np,
        'hr_norm': hr_np,
        'sr_norm': sr_np,
        'psnr': psnr,
        'ssim': ssim,
        'temp_error_mean': temp_error_mean,
        'temp_error_max': temp_error_max,
        'inference_time': inference_time,
        'temp_min': temp_min,
        'temp_max': temp_max
    }


def create_visualization(results: List[Dict], save_dir: str, sample_idx: int):
    """Create visualization for test results"""

    if not SAVE_FILES:
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    result = results[sample_idx]

    # Row 1: Temperature values
    vmin = min(result['lr_temp'].min(), result['hr_temp'].min(), result['sr_temp'].min())
    vmax = max(result['lr_temp'].max(), result['hr_temp'].max(), result['sr_temp'].max())

    im1 = axes[0, 0].imshow(result['lr_temp'], cmap='turbo', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 0].set_title(f'LR Temperature\n{result["lr_temp"].shape}')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    im2 = axes[0, 1].imshow(result['hr_temp'], cmap='turbo', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 1].set_title(f'HR Temperature (GT)\n{result["hr_temp"].shape}')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    im3 = axes[0, 2].imshow(result['sr_temp'], cmap='turbo', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 2].set_title(f'SR Temperature (Pure SwinIR)\n{result["sr_temp"].shape}')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

    # Row 2: Error map and metrics
    error_map = np.abs(result['sr_temp'] - result['hr_temp'])
    im4 = axes[1, 0].imshow(error_map, cmap='hot', aspect='auto')
    axes[1, 0].set_title('Absolute Error (K)')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

    # Hide middle subplot
    axes[1, 1].axis('off')

    # Metrics text
    metrics_text = f"PSNR: {result['psnr']:.2f} dB\n"
    metrics_text += f"SSIM: {result['ssim']:.4f}\n"
    metrics_text += f"Mean Temp Error: {result['temp_error_mean']:.2f} K\n"
    metrics_text += f"Max Temp Error: {result['temp_error_max']:.2f} K\n"
    metrics_text += f"Inference Time: {result['inference_time'] * 1000:.1f} ms"

    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=14,
                    verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    axes[1, 2].axis('off')

    plt.suptitle(f'Pure SwinIR Temperature SR Test - Sample {sample_idx + 1}', fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'test_sample_{sample_idx + 1:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def test_pure_swinir_model(npz_dir: str, model_path: str, num_samples: int = 5,
                           save_dir: str = "./test_swinir_results") -> Dict:
    """Test Pure SwinIR model on multiple samples"""

    # Create output directory only if saving files
    if SAVE_FILES:
        os.makedirs(save_dir, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    logger.info(f"\nLoading Pure SwinIR model from: {model_path}")
    model = load_pure_swinir_model(model_path, device)

    # Create preprocessor
    preprocessor = TemperatureDataPreprocessor(
        target_height=datasets['train']['preprocessor_args']['target_height'],
        target_width=datasets['train']['preprocessor_args']['target_width']
    )

    # Find NPZ files
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files:
        raise ValueError(f"No NPZ files found in {npz_dir}")

    # Use last file (same as training)
    last_file = npz_files[-1]
    logger.info(f"\nUsing last NPZ file: {os.path.basename(last_file)}")

    # Load and test samples
    results = []

    with np.load(last_file, allow_pickle=True) as data:
        # Check data format
        if 'swath_array' in data:
            swath_array = data['swath_array']
        elif 'swaths' in data:
            swath_array = data['swaths']
        else:
            # Single temperature array
            temperature = data['temperature'].astype(np.float32)
            metadata = data['metadata'].item() if hasattr(data['metadata'], 'item') else data['metadata']
            swath_array = [{'temperature': temperature, 'metadata': metadata}]

        total_swaths = len(swath_array)
        logger.info(f"Total swaths in file: {total_swaths}")
        logger.info(f"Testing {num_samples} samples from the end of file")

        # Process from the end of file
        tested_count = 0
        for idx in range(total_swaths - 1, max(0, total_swaths - 100), -1):
            if tested_count >= num_samples:
                break

            try:
                swath = swath_array[idx].item() if hasattr(swath_array[idx], 'item') else swath_array[idx]

                if 'temperature' not in swath:
                    continue

                temperature = swath['temperature'].astype(np.float32)
                metadata = swath.get('metadata', {})
                scale_factor = metadata.get('scale_factor', 1.0)
                temperature = temperature * scale_factor

                # Only fill NaN values if they exist
                if np.any(np.isnan(temperature)):
                    valid_mask = ~np.isnan(temperature)
                    if np.sum(valid_mask) > 0:
                        mean_temp = np.mean(temperature[valid_mask])
                        temperature = np.where(np.isnan(temperature), mean_temp, temperature)
                    else:
                        # If all values are NaN, skip this sample
                        continue

                # Test sample
                logger.info(f"\nTesting sample {tested_count + 1}/{num_samples} (swath {idx})")
                result = test_single_sample(model, temperature, preprocessor, device)

                # Add metadata
                result['swath_index'] = idx
                result['metadata'] = metadata

                results.append(result)
                tested_count += 1

                # Log metrics
                logger.info(f"  PSNR: {result['psnr']:.2f} dB")
                logger.info(f"  SSIM: {result['ssim']:.4f}")
                logger.info(f"  Mean temp error: {result['temp_error_mean']:.2f} K")
                logger.info(f"  Max temp error: {result['temp_error_max']:.2f} K")
                logger.info(f"  Inference time: {result['inference_time'] * 1000:.1f} ms")

            except Exception as e:
                logger.warning(f"Error processing swath {idx}: {e}")
                continue

    # Calculate average metrics
    avg_metrics = {
        'psnr': np.mean([r['psnr'] for r in results]),
        'ssim': np.mean([r['ssim'] for r in results]),
        'temp_error_mean': np.mean([r['temp_error_mean'] for r in results]),
        'temp_error_max': np.mean([r['temp_error_max'] for r in results]),
        'inference_time': np.mean([r['inference_time'] for r in results])
    }

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PURE SWINIR TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model: {os.path.basename(model_path)}")
    logger.info(f"Tested samples: {len(results)}")
    logger.info(f"Average PSNR: {avg_metrics['psnr']:.2f} dB")
    logger.info(f"Average SSIM: {avg_metrics['ssim']:.4f}")
    logger.info(f"Average mean temp error: {avg_metrics['temp_error_mean']:.2f} K")
    logger.info(f"Average max temp error: {avg_metrics['temp_error_max']:.2f} K")
    logger.info(f"Average inference time: {avg_metrics['inference_time'] * 1000:.1f} ms")
    logger.info("=" * 60)

    # Save results if enabled
    if SAVE_FILES:
        # Create visualizations
        for i in range(min(3, len(results))):  # Save first 3 samples
            create_visualization(results, save_dir, i)

        # Save NPZ with all results
        save_path = os.path.join(save_dir, 'test_results.npz')
        np.savez(save_path,
                 results=results,
                 avg_metrics=avg_metrics,
                 model_path=model_path)
        logger.info(f"\nResults saved to: {save_dir}")

    return {
        'results': results,
        'avg_metrics': avg_metrics
    }


def main():
    """Main function"""

    # Configuration - UPDATE THESE PATHS
    NPZ_DIR = "/home/vdidur/temperature_sr_project/data"  # Update based on your sbatch file
    MODEL_PATH = "./experiments/PureTemperatureSR_SwinIR_x2_2nd_generation/models/net_g_20000.pth"  # Update with actual checkpoint
    NUM_SAMPLES = 500
    SAVE_DIR = "./test_swinir_results"

    logger.info("Pure SwinIR Temperature SR Model Testing")
    logger.info(f"Save files: {SAVE_FILES}")

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}")
        logger.info("Looking for available models...")
        experiments_dir = "./experiments"
        if os.path.exists(experiments_dir):
            for root, dirs, files in os.walk(experiments_dir):
                for file in files:
                    if file.startswith("net_g_") and file.endswith(".pth"):
                        logger.info(f"Found: {os.path.join(root, file)}")
        sys.exit(1)

    # Run tests
    try:
        test_results = test_pure_swinir_model(
            npz_dir=NPZ_DIR,
            model_path=MODEL_PATH,
            num_samples=NUM_SAMPLES,
            save_dir=SAVE_DIR
        )

        logger.info("\nTesting completed successfully!")

    except Exception as e:
        logger.error(f"Testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()