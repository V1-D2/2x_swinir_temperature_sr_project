#!/usr/bin/env python3
"""
Cascaded Temperature Super-Resolution for AMSR2 with PureSwinIR Model
Implements three approaches:
1. Model 2x → Model 2x = 4x total
2. Model 2x → Model 2x → Model 2x = 8x total
3. Comparison with bicubic baselines

Key features:
- Multiple cascading strategies for 4x and 8x upscaling
- Unified code style with enhanced logging and visualization
- Temperature statistics tracking
- Processes 5 samples from last NPZ file
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time
import json
from tqdm import tqdm
import cv2
from collections import defaultdict
import warnings
import gc

warnings.filterwarnings('ignore')

# Import model components for PureSwinIR
from swinir_model import PureSwinIRModel
from data_preprocessing import TemperatureDataPreprocessor
from config_temperature import *
from utils import calculate_psnr, calculate_ssim
from basicsr.utils import tensor2img, imwrite

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('cascaded_swinir_temperature_sr.log')
    ]
)
logger = logging.getLogger(__name__)


class PatchBasedPureSwinIRTemperatureSR:
    """Patch-based super-resolution processor for temperature data with Pure SwinIR"""

    def __init__(self, model_path: str, device: torch.device = torch.device('cuda')):
        self.device = device
        self.model = self.load_pure_swinir_model(model_path, device)
        self.preprocessor = TemperatureDataPreprocessor()

    def load_pure_swinir_model(self, model_path: str, device: torch.device) -> PureSwinIRModel:
        """Load trained PureSwinIRModel"""

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

    def create_gaussian_weight_map(self, shape: Tuple[int, int], sigma_ratio: float = 0.3) -> np.ndarray:
        """Create 2D Gaussian weight map for smooth blending"""
        h, w = shape

        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2

        # Calculate Gaussian weights
        sigma_y = h * sigma_ratio
        sigma_x = w * sigma_ratio

        gaussian = np.exp(-((y - center_y) ** 2 / (2 * sigma_y ** 2) +
                            (x - center_x) ** 2 / (2 * sigma_x ** 2)))

        # Normalize to [0, 1]
        gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())

        return gaussian.astype(np.float32)

    def calculate_patch_positions(self, image_shape: Tuple[int, int],
                                  patch_size: Tuple[int, int],
                                  overlap_ratio: float) -> List[Tuple[int, int, int, int]]:
        """Calculate optimal patch positions with adaptive overlap"""
        h, w = image_shape
        ph, pw = patch_size

        # Calculate stride based on overlap
        stride_h = int(ph * (1 - overlap_ratio))
        stride_w = int(pw * (1 - overlap_ratio))

        # Ensure minimum stride
        stride_h = max(1, stride_h)
        stride_w = max(1, stride_w)

        positions = []

        # Calculate positions
        y = 0
        while y < h:
            x = 0
            while x < w:
                # Calculate patch boundaries
                y_end = min(y + ph, h)
                x_end = min(x + pw, w)

                # Adjust start position for edge patches to maintain size
                y_start = max(0, y_end - ph) if y_end == h else y
                x_start = max(0, x_end - pw) if x_end == w else x

                positions.append((y_start, y_end, x_start, x_end))

                # Move to next position
                if x_end >= w:
                    break
                x += stride_w

            if y_end >= h:
                break
            y += stride_h

        return positions

    def calculate_swinir_patch_size(self, input_shape: Tuple[int, int],
                                    target_patch_size: Tuple[int, int] = (1000, 110)) -> Tuple[int, int]:
        """Calculate optimal patch size for SwinIR"""
        h, w = input_shape
        window_size = 8  # SwinIR window size

        # Ensure patch dimensions are divisible by window_size * scale
        factor = window_size * 2  # scale = 2

        patch_h = (target_patch_size[0] // factor) * factor
        patch_w = (target_patch_size[1] // factor) * factor

        # Ensure patches are not larger than input
        patch_h = min(patch_h, h)
        patch_w = min(patch_w, w)

        return (patch_h, patch_w)

    def process_patch(self, patch: np.ndarray) -> np.ndarray:
        """Process single patch through Pure SwinIR model"""
        # Convert to tensor - SwinIR expects [0, 1] range
        patch_tensor = torch.from_numpy(patch).float()
        patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        # Run super-resolution
        with torch.no_grad():
            sr_tensor = self.model.net_g(patch_tensor)
            sr_tensor = torch.clamp(sr_tensor, 0, 1)  # SwinIR outputs in [0, 1]

        # Convert back to numpy
        sr_patch = sr_tensor.cpu().numpy()[0, 0]

        return sr_patch

    def patch_based_super_resolution(self, image: np.ndarray,
                                     patch_size: Tuple[int, int] = (1000, 110),
                                     overlap_ratio: float = 0.75,
                                     stage_name: str = "Stage") -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Apply super-resolution using patch-based approach with weighted blending

        Args:
            image: Input image (temperature in Kelvin)
            patch_size: Size of patches for model
            overlap_ratio: Overlap ratio (0.75 = 75% overlap)
            stage_name: Name for logging

        Returns:
            sr_image: Super-resolution result (temperature in Kelvin)
            stats: Temperature statistics
        """
        start_time = time.time()
        h, w = image.shape
        scale_factor = 2

        logger.info(f"\n{stage_name}: Processing image of size {h}×{w}")

        # Log input temperature statistics
        logger.info(
            f"{stage_name} input - Min: {np.min(image):.2f}K, Max: {np.max(image):.2f}K, Avg: {np.mean(image):.2f}K")

        # Normalize to [0, 1] for SwinIR
        temp_min = np.min(image)
        temp_max = np.max(image)

        if temp_max > temp_min:
            normalized = (image - temp_min) / (temp_max - temp_min)
        else:
            normalized = np.zeros_like(image)

        # Adapt patch size
        patch_size = self.calculate_swinir_patch_size((h, w), patch_size)

        # Initialize output arrays
        output_h, output_w = h * scale_factor, w * scale_factor
        sr_accumulated = np.zeros((output_h, output_w), dtype=np.float64)
        weight_accumulated = np.zeros((output_h, output_w), dtype=np.float64)

        # Create Gaussian weight map for blending
        weight_map = self.create_gaussian_weight_map(
            (patch_size[0] * scale_factor, patch_size[1] * scale_factor)
        )

        # Calculate patch positions
        positions = self.calculate_patch_positions((h, w), patch_size, overlap_ratio)
        logger.info(f"{stage_name}: Created {len(positions)} patches")

        # Process patches with progress bar
        with tqdm(total=len(positions), desc=f"{stage_name} patches") as pbar:
            for i, (y_start, y_end, x_start, x_end) in enumerate(positions):
                # Extract patch
                patch = normalized[y_start:y_end, x_start:x_end]

                # Process patch
                sr_patch = self.process_patch(patch)

                # Calculate output position
                out_y_start = y_start * scale_factor
                out_y_end = y_end * scale_factor
                out_x_start = x_start * scale_factor
                out_x_end = x_end * scale_factor

                # Get weight map for this patch size
                patch_h = out_y_end - out_y_start
                patch_w = out_x_end - out_x_start

                if patch_h != weight_map.shape[0] or patch_w != weight_map.shape[1]:
                    # Create custom weight map for edge patches
                    patch_weight = self.create_gaussian_weight_map((patch_h, patch_w))
                else:
                    patch_weight = weight_map[:patch_h, :patch_w]

                # Accumulate weighted result
                sr_accumulated[out_y_start:out_y_end, out_x_start:out_x_end] += sr_patch * patch_weight
                weight_accumulated[out_y_start:out_y_end, out_x_start:out_x_end] += patch_weight

                # Update progress
                pbar.update(1)

                # Periodic memory cleanup
                if i % 50 == 0:
                    torch.cuda.empty_cache()

        # Normalize by accumulated weights
        mask = weight_accumulated > 0
        sr_normalized = np.zeros_like(sr_accumulated)
        sr_normalized[mask] = sr_accumulated[mask] / weight_accumulated[mask]

        # Denormalize back to temperature
        sr_temperature = sr_normalized * (temp_max - temp_min) + temp_min

        # Calculate statistics
        stats = {
            'min_temp': float(np.min(sr_temperature)),
            'max_temp': float(np.max(sr_temperature)),
            'avg_temp': float(np.mean(sr_temperature)),
            'processing_time': time.time() - start_time,
            'num_patches': len(positions),
            'input_min_temp': float(temp_min),
            'input_max_temp': float(temp_max),
            'input_avg_temp': float(np.mean(image))
        }

        logger.info(
            f"{stage_name} output - Min: {stats['min_temp']:.2f}K, Max: {stats['max_temp']:.2f}K, Avg: {stats['avg_temp']:.2f}K")
        logger.info(f"{stage_name} completed in {stats['processing_time']:.2f}s")

        return sr_temperature, stats


def save_temperature_image(temperature: np.ndarray, save_path: str, dpi: int = 100):
    """Save temperature array as image with exact pixel mapping"""
    # Индивидуальная нормализация для максимального контраста
    temp_norm = (temperature - temperature.min()) / (temperature.max() - temperature.min())
    plt.imsave(save_path, temp_norm, cmap='turbo', origin='upper')


def save_temperature_image_enhanced(temperature: np.ndarray, save_path: str, dpi: int = 100):
    """Save temperature array as image with exact pixel mapping"""
    import matplotlib.cm as cm
    from PIL import Image

    # КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: Используем percentile normalization вместо min/max
    # Это даст более контрастное изображение, игнорируя выбросы
    p_low, p_high = 1, 99  # Как в ZSSR
    temp_min, temp_max = np.percentile(temperature, [p_low, p_high])

    # Clip к percentile диапазону
    temperature_clipped = np.clip(temperature, temp_min, temp_max)

    # Нормализация к [0,1]
    temp_norm = (temperature_clipped - temp_min) / (temp_max - temp_min)

    # Apply turbo colormap and convert to RGB
    turbo_cmap = cm.get_cmap('turbo')
    turbo_rgb = (turbo_cmap(temp_norm)[:, :, :3] * 255).astype(np.uint8)

    # Save as PNG (1:1 pixel mapping)
    Image.fromarray(turbo_rgb).save(save_path)


def cascaded_pure_swinir_temperature_sr(npz_dir: str, model_path: str, num_samples: int = 5,
                                        save_dir: str = "./cascaded_swinir_results") -> List[Dict]:
    """
    Process samples using cascaded Pure SwinIR temperature super-resolution

    Implements:
    - 4x: Model 2x → Model 2x
    - 8x: Model 2x → Model 2x → Model 2x

    Args:
        npz_dir: Directory containing NPZ files
        model_path: Path to trained Pure SwinIR model
        num_samples: Number of samples to process (default: 5)
        save_dir: Directory to save results

    Returns:
        List of results with both 4x and 8x variants
    """
    # Create output directories
    os.makedirs(save_dir, exist_ok=True)
    results_4x_dir = os.path.join(save_dir, 'results_4x')
    results_8x_dir = os.path.join(save_dir, 'results_8x')

    for dir_path in [results_4x_dir, results_8x_dir]:
        os.makedirs(dir_path, exist_ok=True)
        for subdir in ['arrays', 'images', 'visualizations']:
            os.makedirs(os.path.join(dir_path, subdir), exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    # Create patch processor
    logger.info(f"Loading Pure SwinIR model from: {model_path}")
    patch_processor = PatchBasedPureSwinIRTemperatureSR(model_path, device)

    # Find NPZ files
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files:
        raise ValueError(f"No NPZ files found in {npz_dir}")

    # Use last file (same as the other project)
    last_file = npz_files[-1]
    logger.info(f"\nProcessing last NPZ file: {os.path.basename(last_file)}")
    logger.info(f"Processing {num_samples} samples with cascaded super-resolution")

    # Load samples
    results = []
    processed_count = 0

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

        # Process from the end of file (same as the other project)
        for idx in range(total_swaths - 1, max(0, total_swaths - 100), -1):
            if processed_count >= num_samples:
                break

            try:
                swath = swath_array[idx].item() if hasattr(swath_array[idx], 'item') else swath_array[idx]

                if 'temperature' not in swath:
                    continue

                temperature = swath['temperature'].astype(np.float32)
                metadata = swath.get('metadata', {})
                scale_factor = metadata.get('scale_factor', 1.0)
                temperature = temperature * scale_factor

                # Filter invalid values
                temperature = np.where((temperature < 50) | (temperature > 350), np.nan, temperature)
                valid_ratio = np.sum(~np.isnan(temperature)) / temperature.size

                if valid_ratio < 0.5:
                    continue

                # Fill NaN values
                valid_mask = ~np.isnan(temperature)
                if np.sum(valid_mask) > 0:
                    mean_temp = np.mean(temperature[valid_mask])
                    temperature = np.where(np.isnan(temperature), mean_temp, temperature)

                # Store original shape and data
                original_shape = temperature.shape
                original_temp = temperature.copy()

                logger.info(f"\n{'=' * 80}")
                logger.info(f"Processing sample {processed_count + 1}/{num_samples}")
                logger.info(f"Original shape: {original_shape}")

                # === 4x SUPER-RESOLUTION: Model 2x → Model 2x ===
                logger.info(f"\n--- 4x Super-Resolution ---")

                # Stage 1: First 2x
                stage1_sr, stage1_stats = patch_processor.patch_based_super_resolution(
                    original_temp,
                    patch_size=(1000, 110),
                    overlap_ratio=0.75,
                    stage_name="4x Stage 1 (Model 2x)"
                )

                # Stage 2: Second 2x
                stage2_sr, stage2_stats = patch_processor.patch_based_super_resolution(
                    stage1_sr,
                    patch_size=(1000, 110),
                    overlap_ratio=0.75,
                    stage_name="4x Stage 2 (Model 2x)"
                )

                sr_4x = stage2_sr
                logger.info(f"4x Final: {original_shape} → {sr_4x.shape}")

                # === 8x SUPER-RESOLUTION: Model 2x → Model 2x → Model 2x ===
                logger.info(f"\n--- 8x Super-Resolution ---")

                # Stage 3: Third 2x (continuing from 4x result)
                stage3_sr, stage3_stats = patch_processor.patch_based_super_resolution(
                    sr_4x,
                    patch_size=(1000, 110),
                    overlap_ratio=0.75,
                    stage_name="8x Stage 3 (Model 2x)"
                )

                sr_8x = stage3_sr
                logger.info(f"8x Final: {original_shape} → {sr_8x.shape}")

                # Create bicubic baselines
                h, w = original_shape
                bicubic_4x = cv2.resize(original_temp, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
                bicubic_8x = cv2.resize(original_temp, (w * 8, h * 8), interpolation=cv2.INTER_CUBIC)

                # Store results
                result = {
                    'original': original_temp,
                    'sr_4x': sr_4x,
                    'sr_8x': sr_8x,
                    'bicubic_4x': bicubic_4x,
                    'bicubic_8x': bicubic_8x,
                    'stats_4x': {
                        'stage1': stage1_stats,
                        'stage2': stage2_stats,
                        'total_time': stage1_stats['processing_time'] + stage2_stats['processing_time']
                    },
                    'stats_8x': {
                        'stage1': stage1_stats,
                        'stage2': stage2_stats,
                        'stage3': stage3_stats,
                        'total_time': stage1_stats['processing_time'] + stage2_stats['processing_time'] + stage3_stats[
                            'processing_time']
                    },
                    'metadata': {
                        'original_shape': original_shape,
                        'sr_4x_shape': sr_4x.shape,
                        'sr_8x_shape': sr_8x.shape,
                        'swath_index': idx,
                        'scale_factor': metadata.get('scale_factor', 1.0)
                    }
                }

                results.append(result)
                processed_count += 1

                # Log summary
                logger.info(f"\nProcessing Summary:")
                logger.info(f"  4x SR total time: {result['stats_4x']['total_time']:.2f}s")
                logger.info(f"  8x SR total time: {result['stats_8x']['total_time']:.2f}s")

                # Clean memory
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                logger.warning(f"Error processing swath {idx}: {e}")
                continue

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Successfully processed {len(results)} samples")

    # Save results
    save_results(results, results_4x_dir, '4x')
    save_results(results, results_8x_dir, '8x')

    # Create comparison visualizations
    create_comparison_visualizations(results, save_dir)

    # Generate final statistics report
    generate_statistics_report(results, save_dir)

    return results


def save_results(results: List[Dict], save_dir: str, variant: str):
    """Save results for a specific variant (4x or 8x)"""

    arrays_dir = os.path.join(save_dir, 'arrays')
    images_dir = os.path.join(save_dir, 'images')

    for i, result in enumerate(results):
        # Determine which data to save based on variant
        if variant == '4x':
            sr_data = result['sr_4x']
            bicubic_data = result['bicubic_4x']
            stats = result['stats_4x']
        else:  # 8x
            sr_data = result['sr_8x']
            bicubic_data = result['bicubic_8x']
            stats = result['stats_8x']

        # Save arrays
        array_path = os.path.join(arrays_dir, f'sample_{i + 1:03d}.npz')
        np.savez_compressed(
            array_path,
            original=result['original'],
            sr=sr_data,
            bicubic=bicubic_data,
            stats=stats,
            metadata=result['metadata']
        )

        # Save temperature images (colormap)
        save_temperature_image_enhanced(result['original'],
                                        os.path.join(images_dir, f'sample_{i + 1:03d}_original.png'))
        save_temperature_image_enhanced(sr_data,
                                        os.path.join(images_dir, f'sample_{i + 1:03d}_sr_{variant}.png'))
        save_temperature_image_enhanced(bicubic_data,
                                        os.path.join(images_dir, f'sample_{i + 1:03d}_bicubic_{variant}.png'))

        # Save grayscale versions (properly normalized for SwinIR [0,1] range)
        # Percentile normalization для grayscale
        p_low, p_high = 1, 99
        temp_min, temp_max = np.percentile(result['original'], [p_low, p_high])

        # Нормализация с percentile
        def normalize_with_percentile(data):
            data_clipped = np.clip(data, temp_min, temp_max)
            return (data_clipped - temp_min) / (temp_max - temp_min)

        orig_norm = normalize_with_percentile(result['original'])
        sr_norm = normalize_with_percentile(sr_data)
        bicubic_norm = normalize_with_percentile(bicubic_data)

        # Clip to ensure [0, 1] range
        orig_norm = np.clip(orig_norm, 0, 1)
        sr_norm = np.clip(sr_norm, 0, 1)
        bicubic_norm = np.clip(bicubic_norm, 0, 1)

        # Convert to tensors
        orig_tensor = torch.from_numpy(orig_norm).unsqueeze(0).float()
        sr_tensor = torch.from_numpy(sr_norm).unsqueeze(0).float()
        bicubic_tensor = torch.from_numpy(bicubic_norm).unsqueeze(0).float()

        # Convert to images
        orig_img = tensor2img([orig_tensor])
        sr_img = tensor2img([sr_tensor])
        bicubic_img = tensor2img([bicubic_tensor])

        # Save grayscale images
        imwrite(orig_img, os.path.join(images_dir, f'sample_{i + 1:03d}_original_gray.png'))
        imwrite(sr_img, os.path.join(images_dir, f'sample_{i + 1:03d}_sr_{variant}_gray.png'))
        imwrite(bicubic_img, os.path.join(images_dir, f'sample_{i + 1:03d}_bicubic_{variant}_gray.png'))


def create_comparison_visualizations(results: List[Dict], save_dir: str):
    """Create comparison visualizations for both 4x and 8x"""

    # 4x comparison
    create_variant_comparison(results, save_dir, '4x')

    # 8x comparison
    create_variant_comparison(results, save_dir, '8x')

    # Combined comparison showing progression
    create_progression_visualization(results, save_dir)


def create_variant_comparison(results: List[Dict], save_dir: str, variant: str):
    """Create comparison visualization for a specific variant"""

    n_samples = len(results)
    fig, axes = plt.subplots(3, n_samples, figsize=(4 * n_samples, 12))

    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    for i, result in enumerate(results):
        # Original
        axes[0, i].imshow(result['original'], cmap='turbo', aspect='auto')
        axes[0, i].set_title(f'Original {i + 1}\n{result["original"].shape}')
        axes[0, i].axis('off')

        # SR result
        if variant == '4x':
            sr_data = result['sr_4x']
            stats = result['stats_4x']
        else:
            sr_data = result['sr_8x']
            stats = result['stats_8x']

        axes[1, i].imshow(sr_data, cmap='turbo', aspect='auto')
        axes[1, i].set_title(f'SR {variant}\nTime: {stats["total_time"]:.1f}s')
        axes[1, i].axis('off')

        # Bicubic baseline
        bicubic_data = result[f'bicubic_{variant}']
        axes[2, i].imshow(bicubic_data, cmap='turbo', aspect='auto')
        axes[2, i].set_title(f'Bicubic {variant}\n{bicubic_data.shape}')
        axes[2, i].axis('off')

    # Calculate average times
    avg_time = np.mean([r[f'stats_{variant}']['total_time'] for r in results])

    title = f'{variant} Super-Resolution Comparison ({n_samples} samples)\n'
    title += f'Average processing time: {avg_time:.1f}s'

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{variant}_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_progression_visualization(results: List[Dict], save_dir: str):
    """Create visualization showing progression from original to 8x"""

    if len(results) == 0:
        return

    # Use first sample for detailed progression
    result = results[0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Temperature maps
    images = [
        (result['original'], 'Original', result['original'].shape),
        (result['sr_4x'], '4x SR (2x→2x)', result['sr_4x'].shape),
        (result['sr_8x'], '8x SR (2x→2x→2x)', result['sr_8x'].shape)
    ]

    for i, (img, title, shape) in enumerate(images):
        # Индивидуальная нормализация каждого изображения
        # Используем percentile normalization для каждого изображения
        p_low, p_high = 1, 99
        img_min, img_max = np.percentile(img, [p_low, p_high])
        img_clipped = np.clip(img, img_min, img_max)
        img_norm = (img_clipped - img_min) / (img_max - img_min)
        im = axes[0, i].imshow(img_norm, cmap='turbo', aspect='auto')
        axes[0, i].set_title(f'{title}\n{shape}')
        axes[0, i].axis('off')
        plt.colorbar(im, ax=axes[0, i], fraction=0.046)

    # Row 2: Zoomed regions for detail comparison
    if result['sr_8x'].shape[0] >= 512:
        h, w = result['sr_8x'].shape
        center_y, center_x = h // 2, w // 2
        size = 256

        y1 = max(0, center_y - size)
        y2 = min(h, center_y + size)
        x1 = max(0, center_x - size)
        x2 = min(w, center_x + size)

        # Original zoomed (scaled to match 8x)
        orig_h, orig_w = result['original'].shape
        orig_y1, orig_y2 = y1 // 8, y2 // 8
        orig_x1, orig_x2 = x1 // 8, x2 // 8
        orig_zoom = cv2.resize(result['original'][orig_y1:orig_y2, orig_x1:orig_x2],
                               (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)

        # 4x zoomed (scaled to match 8x)
        sr4x_y1, sr4x_y2 = y1 // 2, y2 // 2
        sr4x_x1, sr4x_x2 = x1 // 2, x2 // 2
        sr4x_zoom = cv2.resize(result['sr_4x'][sr4x_y1:sr4x_y2, sr4x_x1:sr4x_x2],
                               (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)

        zoomed_images = [
            (orig_zoom, 'Original (8x upscaled)'),
            (sr4x_zoom, '4x SR (2x upscaled)'),
            (result['sr_8x'][y1:y2, x1:x2], '8x SR')
        ]

        for i, (img, title) in enumerate(zoomed_images):
            # Используем percentile normalization для каждого изображения
            p_low, p_high = 1, 99
            img_min, img_max = np.percentile(img, [p_low, p_high])
            img_clipped = np.clip(img, img_min, img_max)
            img_norm = (img_clipped - img_min) / (img_max - img_min)
            im = axes[1, i].imshow(img_norm, cmap='turbo', aspect='auto')
            axes[1, i].set_title(f'{title} - Zoomed')
            axes[1, i].axis('off')
    else:
        for i in range(3):
            axes[1, i].axis('off')

    # Add temperature statistics
    stats_text = "Temperature Statistics (K):\n\n"
    stats_text += f"Original: [{np.min(result['original']):.1f}, {np.max(result['original']):.1f}], "
    stats_text += f"Avg: {np.mean(result['original']):.1f}\n"
    stats_text += f"4x SR: [{np.min(result['sr_4x']):.1f}, {np.max(result['sr_4x']):.1f}], "
    stats_text += f"Avg: {np.mean(result['sr_4x']):.1f}\n"
    stats_text += f"8x SR: [{np.min(result['sr_8x']):.1f}, {np.max(result['sr_8x']):.1f}], "
    stats_text += f"Avg: {np.mean(result['sr_8x']):.1f}"

    fig.text(0.02, 0.02, stats_text, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

    plt.suptitle('Temperature Super-Resolution Progression', fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'progression_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_statistics_report(results: List[Dict], save_dir: str):
    """Generate comprehensive statistics report"""

    report_path = os.path.join(save_dir, 'statistics_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CASCADED PURE SWINIR TEMPERATURE SUPER-RESOLUTION STATISTICS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total samples processed: {len(results)}\n")
        f.write(f"Model: Pure SwinIR-based Temperature SR\n")
        f.write(f"Processing variants: 4x (2x→2x) and 8x (2x→2x→2x)\n\n")

        # Individual sample statistics
        f.write("INDIVIDUAL SAMPLE STATISTICS\n")
        f.write("-" * 80 + "\n")

        for i, result in enumerate(results):
            f.write(f"\nSample {i + 1}:\n")
            f.write(f"  Original shape: {result['metadata']['original_shape']}\n")
            f.write(f"  Original temperature: Min={np.min(result['original']):.2f}K, "
                    f"Max={np.max(result['original']):.2f}K, "
                    f"Avg={np.mean(result['original']):.2f}K\n")

            # 4x stats
            f.write(f"\n  4x Super-Resolution:\n")
            f.write(f"    Final shape: {result['metadata']['sr_4x_shape']}\n")
            f.write(f"    Temperature: Min={np.min(result['sr_4x']):.2f}K, "
                    f"Max={np.max(result['sr_4x']):.2f}K, "
                    f"Avg={np.mean(result['sr_4x']):.2f}K\n")
            f.write(f"    Processing time: {result['stats_4x']['total_time']:.2f}s\n")

            # 8x stats
            f.write(f"\n  8x Super-Resolution:\n")
            f.write(f"    Final shape: {result['metadata']['sr_8x_shape']}\n")
            f.write(f"    Temperature: Min={np.min(result['sr_8x']):.2f}K, "
                    f"Max={np.max(result['sr_8x']):.2f}K, "
                    f"Avg={np.mean(result['sr_8x']):.2f}K\n")
            f.write(f"    Processing time: {result['stats_8x']['total_time']:.2f}s\n")

        # Summary statistics
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")

        # Average processing times
        avg_4x_time = np.mean([r['stats_4x']['total_time'] for r in results])
        avg_8x_time = np.mean([r['stats_8x']['total_time'] for r in results])

        f.write(f"\nAverage processing times:\n")
        f.write(f"  4x SR: {avg_4x_time:.2f}s\n")
        f.write(f"  8x SR: {avg_8x_time:.2f}s\n")

        # Temperature preservation analysis
        f.write(f"\nTemperature preservation analysis:\n")

        temp_diffs_4x = []
        temp_diffs_8x = []

        for result in results:
            orig_avg = np.mean(result['original'])
            sr4x_avg = np.mean(result['sr_4x'])
            sr8x_avg = np.mean(result['sr_8x'])

            temp_diffs_4x.append(abs(sr4x_avg - orig_avg))
            temp_diffs_8x.append(abs(sr8x_avg - orig_avg))

        f.write(f"  4x SR average temperature difference: {np.mean(temp_diffs_4x):.2f}K\n")
        f.write(f"  8x SR average temperature difference: {np.mean(temp_diffs_8x):.2f}K\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Report generated successfully\n")

    logger.info(f"Statistics report saved to: {report_path}")


def main():
    """Main function for cascaded Pure SwinIR temperature SR"""
    import argparse

    parser = argparse.ArgumentParser(description='Cascaded Pure SwinIR Temperature Super-Resolution')
    parser.add_argument('--npz-dir', type=str, required=True,
                        help='Directory containing NPZ files')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained Pure SwinIR model checkpoint')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to process (default: 5)')
    parser.add_argument('--save-dir', type=str, default='./cascaded_swinir_results',
                        help='Directory to save results')
    parser.add_argument('--overlap-ratio', type=float, default=0.75,
                        help='Overlap ratio for patches (default: 0.75)')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("CASCADED PURE SWINIR TEMPERATURE SUPER-RESOLUTION")
    logger.info("=" * 80)
    logger.info(f"NPZ directory: {args.npz_dir}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Save directory: {args.save_dir}")
    logger.info("=" * 80)

    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)

    # Process samples
    try:
        results = cascaded_pure_swinir_temperature_sr(
            npz_dir=args.npz_dir,
            model_path=args.model_path,
            num_samples=args.num_samples,
            save_dir=args.save_dir
        )

        logger.info("\n" + "=" * 80)
        logger.info("CASCADED PROCESSING COMPLETED SUCCESSFULLY!")
        logger.info(f"Processed {len(results)} samples with 4x and 8x upscaling")
        logger.info(f"Results saved to: {args.save_dir}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()