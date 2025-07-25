import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.base_model import BaseModel
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils import tensor2img, imwrite
from collections import OrderedDict
import numpy as np
import os.path as osp
from tqdm import tqdm

# Import our modified SwinIR - NO CHANGES TO THIS IMPORT
from models.network_swinir import SwinIR
from utils import calculate_psnr, calculate_ssim


class TemperaturePerceptualLoss(nn.Module):
    """Perceptual loss adapted for temperature data - KEPT UNCHANGED"""

    def __init__(self, feature_weights=None):
        super().__init__()
        if feature_weights is None:
            self.feature_weights = [1.0, 1.0, 1.0, 1.0]
        else:
            self.feature_weights = feature_weights

        # Create simple feature extractor for temperature data
        self.feature_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 2, 1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 2, 1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, 2, 1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 2, 1),
                nn.ReLU(inplace=True)
            )
        ])

        # Freeze weights for stability
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        """Calculate perceptual loss between x and y"""
        loss = 0

        feat_x = x
        feat_y = y

        for i, layer in enumerate(self.feature_extractor):
            feat_x = layer(feat_x)
            feat_y = layer(feat_y)

            # L1 loss between features
            loss += self.feature_weights[i] * F.l1_loss(feat_x, feat_y)

        return loss


class PhysicsConsistencyLoss(nn.Module):
    """Loss for maintaining physical consistency of temperature data - KEPT UNCHANGED"""

    def __init__(self, gradient_weight=0.1, smoothness_weight=0.05):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.smoothness_weight = smoothness_weight

    def forward(self, pred, target):
        """
        Calculate loss with physical properties of temperature field
        """
        # Main L1 loss
        main_loss = F.l1_loss(pred, target)

        # Gradient loss - preserve edge sharpness
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        gradient_loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

        # Smoothness loss - avoid artifacts
        smooth_x = pred[:, :, :, 1:] - 2 * pred[:, :, :, :-1] + pred[:, :, :, :-1]
        smooth_y = pred[:, :, 1:, :] - 2 * pred[:, :, :-1, :] + pred[:, :, :-1, :]
        smoothness_loss = torch.mean(torch.abs(smooth_x)) + torch.mean(torch.abs(smooth_y))

        total_loss = main_loss + self.gradient_weight * gradient_loss + self.smoothness_weight * smoothness_loss

        return total_loss, {
            'main': main_loss,
            'gradient': gradient_loss,
            'smoothness': smoothness_loss
        }


@MODEL_REGISTRY.register()
class PureSwinIRModel(BaseModel):
    """Pure SwinIR model for Temperature Super-Resolution without GAN"""

    def __init__(self, opt):
        super(PureSwinIRModel, self).__init__(opt)

        # Build SwinIR generator - EXACTLY THE SAME AS BEFORE
        self.net_g = self.build_swinir_generator(opt)
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # Initialize training settings
        if self.is_train:
            self.init_training_settings()

    def build_swinir_generator(self, opt):
        """Build SwinIR generator for temperature data - KEPT UNCHANGED"""
        opt_net = opt['network_g']

        # Parameters for 2x upscaling with 1 channel
        model = SwinIR(
            upscale=2,
            in_chans=1,  # 1 channel for temperature data
            img_size=opt_net.get('img_size', 64),
            window_size=opt_net.get('window_size', 8),
            img_range=1.,
            depths=opt_net.get('depths', [6, 6, 6, 6]),
            # depths=opt_net.get('depths', [6, 6, 6, 6, 6, 6]),
            embed_dim=opt_net.get('embed_dim', 60),
            num_heads=opt_net.get('num_heads', [6, 6, 6, 6]),
            # num_heads=opt_net.get('num_heads', [6, 6, 6, 6, 6, 6]),
            mlp_ratio=opt_net.get('mlp_ratio', 4),
            upsampler=opt_net.get('upsampler', 'pixelshuffle'),
            resi_connection=opt_net.get('resi_connection', '3conv')
        )

        return model

    def init_training_settings(self):
        """Initialize training settings with physical losses only"""
        self.net_g.train()
        train_opt = self.opt['train']

        print(f"DEBUG: train_opt keys: {list(train_opt.keys())}")
        print(f"DEBUG: pixel_opt exists: {'pixel_opt' in train_opt}")
        print(f"DEBUG: pixel_opt value: {train_opt.get('pixel_opt')}")

        # Setup pixel loss
        if train_opt.get('pixel_opt'):
            self.cri_pix = PhysicsConsistencyLoss(
                gradient_weight=train_opt['pixel_opt'].get('gradient_weight', 0.1),
                smoothness_weight=train_opt['pixel_opt'].get('smoothness_weight', 0.05)
            )
        else:
            self.cri_pix = None

        # Setup perceptual loss for temperatures
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = TemperaturePerceptualLoss(
                feature_weights=train_opt['perceptual_opt'].get('feature_weights', [1.0, 1.0, 1.0, 1.0])
            ).to(self.device)
        else:
            self.cri_perceptual = None

        # Setup optimizers - ONLY FOR GENERATOR
        self.setup_optimizers()
        # Setup scheduler manually since CosineAnnealingLR is not in BasicSR
        train_opt = self.opt['train']
        if 'scheduler' in train_opt:
            scheduler_type = train_opt['scheduler']['type']
            if scheduler_type == 'CosineAnnealingLR':
                from torch.optim.lr_scheduler import CosineAnnealingLR
                self.schedulers.append(
                    CosineAnnealingLR(
                        self.optimizer_g,
                        T_max=train_opt['scheduler']['T_max'],
                        eta_min=train_opt['scheduler']['eta_min']
                    )
                )
            elif scheduler_type == 'MultiStepLR':
                from torch.optim.lr_scheduler import MultiStepLR
                self.schedulers.append(
                    MultiStepLR(
                        self.optimizer_g,
                        milestones=train_opt['scheduler']['milestones'],
                        gamma=train_opt['scheduler']['gamma']
                    )
                )
            else:
                print(f"Warning: Scheduler {scheduler_type} not implemented, training without scheduler")
        else:
            print("No scheduler configured, training with constant learning rate")

    def setup_optimizers(self):
        """Set up optimizer for generator only"""
        train_opt = self.opt['train']

        # Optimizer for generator
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        """Feed data - KEPT UNCHANGED"""
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        """Optimization without GAN components"""
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()

        # Pixel loss - ALWAYS
        if self.cri_pix:
            l_g_pix, pix_losses = self.cri_pix(self.output, self.gt)
            l_g_total += l_g_pix * self.opt['train']['pixel_opt']['loss_weight']
            loss_dict['l_g_pix'] = l_g_pix
            for k, v in pix_losses.items():
                loss_dict[f'l_g_pix_{k}'] = v

        # Perceptual loss
        if self.cri_perceptual and current_iter % 5 == 0:
            l_g_percep = self.cri_perceptual(self.output, self.gt)
            l_g_total += l_g_percep * self.opt['train']['perceptual_opt']['loss_weight']
            loss_dict['l_g_percep'] = l_g_percep

        # Ensure we have at least one loss
        if isinstance(l_g_total, int) and l_g_total == 0:
            # Fallback to simple L1 loss if no other losses are computed
            l_g_total = torch.nn.functional.l1_loss(self.output, self.gt)
            loss_dict['l_g_fallback'] = l_g_total

        l_g_total.backward()

        # Gradient clipping
        if self.opt['train'].get('use_grad_clip', True):
            torch.nn.utils.clip_grad_norm_(
                self.net_g.parameters(),
                max_norm=self.opt['train'].get('grad_clip_norm', 7.0)
            )

        self.optimizer_g.step()

        # Calculate PSNR and SSIM metrics
        if current_iter % 500 == 0:
            with torch.no_grad():
                output_clamped = torch.clamp(self.output, 0, 1)

                try:
                    pred_np = tensor2img([output_clamped])
                    gt_np = tensor2img([self.gt])

                    # Calculate PSNR
                    psnr_value = calculate_psnr(pred_np, gt_np, crop_border=0, test_y_channel=False)
                    loss_dict['psnr'] = psnr_value

                    # Calculate SSIM
                    ssim_value = calculate_ssim(pred_np, gt_np, crop_border=0, test_y_channel=False)
                    loss_dict['ssim'] = ssim_value

                except Exception as e:
                    loss_dict['psnr'] = 0.0
                    loss_dict['ssim'] = 0.0
                    print(f"Error calculating metrics: {e}")

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # Clear cache periodically
        if current_iter % 10 == 0:
            torch.cuda.empty_cache()

    def test(self):
        """Test with physical constraints preservation"""
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
            # Clamp for physical correctness
            self.output = torch.clamp(self.output, 0, 1)
        self.net_g.train()

    def get_current_visuals(self):
        """Get current visuals - KEPT UNCHANGED"""
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        """Save models and training states - SIMPLIFIED"""
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Validation with metrics for temperature data - KEPT UNCHANGED"""
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            self._initialize_best_metric_results(dataset_name)

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img

            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img

            # Save images
            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             f'{current_iter}_{dataset_name}',
                                             f'{idx:08d}.png')
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             dataset_name,
                                             f'{idx:08d}.png')

                imwrite(sr_img, save_img_path)

            # Calculate metrics
            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self._update_metric(metric_data, dataset_name, name, opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {idx:08d}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            self._report_metric_results(dataset_name)

    def _initialize_best_metric_results(self, dataset_name):
        """Initialize metric results dict - KEPT UNCHANGED"""
        if not hasattr(self, 'best_metric_results'):
            self.best_metric_results = {}

        record = {}
        for metric, content in self.opt['val']['metrics'].items():
            record[metric] = {'better': 'higher', 'val': float('-inf'), 'iter': -1}
            if content.get('better', 'higher') == 'lower':
                record[metric]['better'] = 'lower'
                record[metric]['val'] = float('inf')
        self.best_metric_results[dataset_name] = record

    def _update_metric(self, metric_data, dataset_name, metric_name, opt_):
        """Update metric results - KEPT UNCHANGED"""
        if metric_name == 'psnr':
            from basicsr.metrics import calculate_psnr
            value = calculate_psnr(metric_data['img'], metric_data['img2'],
                                   crop_border=opt_.get('crop_border', 0),
                                   test_y_channel=opt_.get('test_y_channel', False))
        elif metric_name == 'ssim':
            from basicsr.metrics import calculate_ssim
            value = calculate_ssim(metric_data['img'], metric_data['img2'],
                                   crop_border=opt_.get('crop_border', 0),
                                   test_y_channel=opt_.get('test_y_channel', False))
        else:
            from basicsr.metrics import calculate_metric
            value = calculate_metric(metric_data, opt_)

        if not hasattr(self, 'metric_results'):
            self.metric_results = {}
        if dataset_name not in self.metric_results:
            self.metric_results[dataset_name] = {}
        self.metric_results[dataset_name][metric_name] = value

        if value > self.best_metric_results[dataset_name][metric_name]['val']:
            self.best_metric_results[dataset_name][metric_name]['val'] = value
            self.best_metric_results[dataset_name][metric_name]['iter'] = metric_data.get('iter', -1)

    def _report_metric_results(self, dataset_name):
        """Report average metrics - KEPT UNCHANGED"""
        from basicsr.utils import get_root_logger
        logger = get_root_logger()

        if hasattr(self, 'metric_results') and dataset_name in self.metric_results:
            logger.info(f'\n=== Validation Results for {dataset_name} ===')

            for metric_name, metric_value in self.metric_results[dataset_name].items():
                if metric_name == 'psnr':
                    logger.info(f'PSNR: {metric_value:.2f} dB')
                elif metric_name == 'ssim':
                    logger.info(f'SSIM: {metric_value:.4f}')
                else:
                    logger.info(f'{metric_name.upper()}: {metric_value:.4f}')

                if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
                    best_info = self.best_metric_results[dataset_name][metric_name]
                    if metric_name == 'psnr':
                        logger.info(f'Best PSNR: {best_info["val"]:.2f} dB at iteration {best_info["iter"]}')
                    elif metric_name == 'ssim':
                        logger.info(f'Best SSIM: {best_info["val"]:.4f} at iteration {best_info["iter"]}')
                    else:
                        logger.info(f'Best {metric_name}: {best_info["val"]:.4f} at iteration {best_info["iter"]}')

            logger.info('=' * 60)