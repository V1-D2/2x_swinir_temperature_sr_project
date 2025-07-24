# Configuration for training pure SwinIR Temperature Super-Resolution model

# General parameters
name = 'PureTemperatureSR_SwinIR_x2'
model_type = 'PureSwinIRModel'  # Changed from TemperatureSRModel
scale = 2
num_gpu = 1  # Number of GPUs

# Data parameters - UNCHANGED
datasets = {
    'train': {
        'name': 'TemperatureTrainDataset',
        'dataroot_gt': None,  # Will be set in train script
        'npz_files': [],  # Will be set in train script
        'preprocessor_args': {
            'target_height': 2000,
            'target_width': 220
        },
        'scale_factor': 2,
        'batch_size': 2,
        'samples_per_file': 1000,  # Memory management limit
        'num_worker': 8,
        'pin_memory': True,
        'persistent_workers': True
    },
    'val': {
        'name': 'TemperatureValDataset',
        'dataroot_gt': None,
        'npz_file': None,  # Will be set in train script
        'n_samples': 100,
        'scale_factor': 2
    }
}

# Network parameters - ONLY GENERATOR, DISCRIMINATOR REMOVED
network_g = {
    'type': 'SwinIR',
    'upscale': 2,
    'in_chans': 1,  # Temperature data - 1 channel
    'img_size': 64,
    'window_size': 8,
    'img_range': 1.,
    'depths': [6, 6, 6, 6, 6, 6],
    'embed_dim': 60,
    'num_heads': [6, 6, 6, 6, 6, 6],
    'mlp_ratio': 4,
    'upsampler': 'pixelshuffle',
    'resi_connection': '3conv'
}

# REMOVED network_d configuration

# Path settings - UNCHANGED
path = {
    'pretrain_network_g': None,
    'strict_load_g': True,
    'resume_state': None,
    'root': './',
    'experiments_root': './experiments',
    'models': './experiments/models',
    'training_states': './experiments/training_states',
    'log': './experiments/log',
    'visualization': './experiments/visualization'
}

# Training parameters - REMOVED GAN-RELATED SETTINGS
train = {
    'ema_decay': 0.999,
    'optim_g': {
        'type': 'Adam',
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'betas': [0.9, 0.99]
    },
    # REMOVED optim_d
    '''
    'scheduler': {
        'type': 'CosineAnnealingLR',
        'T_max': 100000,
        'eta_min': 1e-6
    },
    '''
    'scheduler': {
        'type': 'MultiStepLR',  # This would work
        'milestones': [50000, 100000],
        'gamma': 0.5
    },
    # Loss functions - REMOVED gan_opt
    'pixel_opt': {
        'type': 'PhysicsConsistencyLoss',
        'loss_weight': 100.0,
        'gradient_weight': 0.08,
        'smoothness_weight': 0.03,
        'reduction': 'mean'
    },
    'perceptual_opt': {
        'type': 'TemperaturePerceptualLoss',
        'loss_weight': 10.0,
        'feature_weights': [0.1, 0.2, 1.0, 1.0]
    },
    # REMOVED gan_opt
    # REMOVED net_d_iters and net_d_init_iters
    'manual_seed': 10,
    'use_grad_clip': True,
    'grad_clip_norm': 7.0,
    'use_ema': True  # Exponential Moving Average
}

# Validation parameters - UNCHANGED
val = {
    'val_freq': 30000,
    'save_img': True,
    'metrics': {
        'psnr': {
            'type': 'calculate_psnr',
            'crop_border': 0,
            'test_y_channel': False
        },
        'ssim': {
            'type': 'calculate_ssim',
            'crop_border': 0,
            'test_y_channel': False
        }
    }
}

# Logging - UNCHANGED
logger = {
    'print_freq': 1000,
    'save_checkpoint_freq': 400000,
    'use_tb_logger': False,
    'wandb': {
        'project': 'pure-temperature-sr',  # Changed project name
        'resume_id': None
    }
}

# Distributed training - UNCHANGED
dist_params = {
    'backend': 'nccl',
    'port': 29500
}

# Temperature-specific parameters - UNCHANGED
temperature_specific = {
    'preserve_relative_values': True,
    'temperature_range': [80, 400],  # Kelvin
    'physical_constraints': {
        'enforce_smoothness': True,
        'preserve_gradients': True,
        'max_gradient': 20.0  # Maximum temperature gradient
    }
}

# Incremental training - UNCHANGED
incremental_training = {
    'enabled': True,
    'epochs_per_file': 1,
    'learning_rate_decay_per_file': 1.0,
    'checkpoint_per_file': False,
    'shuffle_files': True
}

# Additional parameters - UNCHANGED
others = {
    'use_amp': False,  # Automatic Mixed Precision
    'num_threads': 8,
    'seed': 10
}