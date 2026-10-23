#!/usr/bin/env python3
"""
Training script for unconditional diffusion on MIROC6 data.

Following Rozet et al. 2023 "Score-based Data Assimilation" approach:
- Train a score network s(x, σ) ≈ ∇ log p_σ(x) to learn the prior distribution
- No conditioning on external data (SST, labels, etc.)
- Later use for data assimilation by combining with observation likelihood

Key differences from cBottle's default training:
1. Unconditional: No SST input, no labels
2. 2-channel output: Only tas and sst fields
3. Full dropout masking: 100% unconditional for video mode
4. Monthly time scale: 30-day steps, 12-month windows
"""

import sys
import os
from pathlib import Path
import dataclasses
import logging
import time
import warnings
import subprocess
import csv
import threading
import copy
import pickle

import numpy as np
import torch
from tqdm import tqdm

# Suppress the NCHW to NHWC warnings (they're just informational)
warnings.filterwarnings('ignore', message='Cannot do a zero-copy NCHW to NHWC')


def get_gpu_stats():
    """Get GPU utilization and memory stats."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 3:
                gpu_util = int(parts[0])
                mem_used = int(parts[1])
                mem_total = int(parts[2])
                return gpu_util, mem_used, mem_total
    except Exception:
        pass
    return None, None, None

# Add src and scripts to path to allow imports - MUST be before any cbottle imports
CURRENT_DIR = Path(__file__).parent.absolute()
SRC_DIR = CURRENT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(CURRENT_DIR))

# Force reload of cbottle modules from local source
import importlib
if 'cbottle' in sys.modules:
    # Remove all cbottle modules to force reimport from local source
    to_remove = [k for k in sys.modules.keys() if k.startswith('cbottle')]
    for k in to_remove:
        del sys.modules[k]

from cbottle.datasets.base import BatchInfo, TimeUnit

# Debug: print where cbottle is loaded from
import cbottle.models
from cbottle.datasets import dataset_2d
from cbottle.dataclass_parser import parse_args
from cbottle import distributed as dist
from cbottle.config.training.masking import MaskingConfig, StrategyWeights
import cbottle.config.models

# Import the default training loop
try:
    import training_loop_base
except ImportError:
    from scripts import training_loop_base

# Surface-only MIROC6 loader is not shipped in this package.
# Profiles training (train_ocean_prior.py) overrides get_dataset / _get_batch_info.
_gen_module = None


def unconditional_masking_config() -> MaskingConfig:
    """
    Masking config for unconditional generation.
    100% full_dropout = no frames are conditioned, pure unconditional diffusion.
    """
    return MaskingConfig(
        strategy_weights=StrategyWeights(
            random=0.0,
            blockwise=0.0,
            interpolation=0.0,
            full_dropout=1.0,  # Always unconditional
        ),
        random_mask_prob=1.0,
        block_mask_fraction=1.0,
        block_predict_past=False,
        interpolation_mask_fraction=1.0,
    )



@dataclasses.dataclass
class MirocTrainingLoop(training_loop_base.TrainingLoop):
    """
    Training loop for unconditional diffusion on MIROC6 data.
    
    Follows Rozet et al. 2023 approach:
    - Learn score function s(x, σ) for the prior p(x)
    - No external conditioning (SST, labels)
    - Pure denoising score matching
    """
    
    # MIROC specific settings
    miroc_zarr_path: str = "/path/to/miroc6_profiles.zarr"
    land_mask_climatology_path: str | None = None  # pickle with land_mask["tos"]
    time_window_months: int = 12
    
    # Override defaults for unconditional training
    with_era5: bool = False          # No ERA5 mixing
    use_labels: bool = False         # No dataset labels
    monthly_sst_input: bool = False  # No SST conditioning
    variables: str = "miroc"         # Use 2-channel miroc config
    
    # Network settings for smaller model (2 channels vs 37)
    network: training_loop_base.SongUnetConfig = dataclasses.field(
        default_factory=lambda: training_loop_base.SongUnetConfig(
            model_channels=64,  # Smaller model for 2 channels
            include_legacy_calendar_bug=False,
        )
    )
    
    # Use unconditional masking
    masking_config: MaskingConfig = dataclasses.field(
        default_factory=unconditional_masking_config
    )
    plot_every_steps: int | None = None
    generate_every_steps: int | None = None
    generate_num_samples: int = 2
    generate_num_steps: int = 50
    max_inline_generators: int = 1
    skip_log_likelihood: bool = True

    @property
    def variable_config(self):
        """Return MIROC6 variable configuration (2 channels: tas, sst)."""
        from cbottle.datasets.dataset_3d import VARIABLE_CONFIGS
        return VARIABLE_CONFIGS["miroc"]

    def _get_batch_info(self) -> BatchInfo:
        """Get batch info for MIROC6 data (2 channels: tas, sst)."""
        raise NotImplementedError(
            "Surface MIROC6 dataloader is not included. "
            "Use train_ocean_prior.py for profiles training."
        )
    
    @property
    def out_channels(self):
        """Override to return 2 channels (tas, sst)."""
        return 2  # Hardcode to avoid cached_property issues
    
    @property
    def model_config(self) -> cbottle.config.models.ModelConfigV1:
        """
        Build model config for unconditional diffusion.
        
        Key settings:
        - label_dim=0: No dataset labels
        - condition_channels: Only mask channel for video, 0 for image
        - out_channels=2: Only tas and sst
        """
        out_channels = self.out_channels  # 2 for miroc config
        
        # Use time_window_months since time_length may not be set yet
        time_len = self.time_window_months
        is_video = time_len > 1
        
        
        if is_video:
            # Video mode: condition on mask + previous predictions
            # For unconditional: still need mask channel to indicate which frames to predict
            condition_channels = out_channels + 1  # channels + mask
            config = cbottle.config.models.ModelConfigV1(
                label_dim=0,  # No labels
                out_channels=out_channels,
                condition_channels=condition_channels,
                model_channels=self.network.model_channels,
                time_length=time_len,
                label_dropout=0.0,
                calendar_include_legacy_bug=self.network.include_legacy_calendar_bug,
            )
            return config
        else:
            # Image mode: truly unconditional
            config = cbottle.config.models.ModelConfigV1(
                label_dim=0,
                out_channels=out_channels,
                condition_channels=0,  # No conditioning
                model_channels=self.network.model_channels,
                calendar_include_legacy_bug=self.network.include_legacy_calendar_bug,
                enable_classifier=False,
            )
            return config

    def get_dataset(self, train: bool):
        """
        Get MIROC6 dataset for unconditional training.
        """
        raise NotImplementedError(
            "Surface MIROC6 dataloader is not included. "
            "Use train_ocean_prior.py for profiles training."
        )

    def get_network(self) -> torch.nn.Module:
        """Override to ensure correct model config is used."""
        import cbottle.models
        config = self.model_config
        return cbottle.models.get_model(config)
    
    def get_loss_fn(self):
        """Override to use sigma_data=0.5 to match dataset and model.
        
        The MIROC6 dataset loader scales data by SIGMA_DATA=0.5, and the
        EDMPrecond model uses sigma_data=0.5 by default. The parent class
        hardcodes sigma_data=1.0, which causes a mismatch in the loss weighting.
        """
        import cbottle.loss
        return cbottle.loss.EDMLoss(
            P_mean=self.p_mean,
            P_std=self.p_std,
            sigma_data=0.5,  # Match dataset scaling and model default
            sigma_max=self.sigma_max,
            sigma_min=self.sigma_min,
            distribution=self.noise_distribution,
            classifier_weight=self.ibtracs_loss_weight,
        )
    
    
    def setup(self):
        """Initialize training loop with MIROC6 settings."""
        # Set time_length from our custom setting
        self.time_length = self.time_window_months
        
        # Clear the cached batch_info from parent class and set our own
        # The parent uses @functools.cached_property which we need to override
        if 'batch_info' in self.__dict__:
            del self.__dict__['batch_info']
        # Set batch_info directly as instance attribute (overrides cached_property)
        object.__setattr__(self, 'batch_info', self._get_batch_info())
        
        super().setup()
        self._inline_generators: list[threading.Thread] = []
    
    def train_step(self, *, target, timestamp=None, **batch):
        """Override train_step to mask out land pixels for TOS channel."""
        from cbottle import training_stats
        
        # Compute loss as normal
        loss: cbottle.loss.Output = self.loss_fn(
            self._curry_net(self.ddp, batch),
            target,
            classifier_labels=batch.get("classifier_labels"),
        )
        
        # Get land mask from climatology file (not from condition)
        # Load it once and cache it
        if not hasattr(self, '_land_mask_tensor'):
            import pickle
            import os
            clim_path = self.land_mask_climatology_path
            if clim_path and os.path.exists(clim_path):
                with open(clim_path, 'rb') as f:
                    clim_data = pickle.load(f)
                land_mask_np = clim_data['land_mask']['tos']  # Boolean array
                # Reorder to match target pixel ordering (HEALPIX_PAD_XY)
                import earth2grid
                land_mask_torch = torch.as_tensor(land_mask_np, dtype=torch.float32)
                land_mask_torch = earth2grid.healpix.reorder(
                    land_mask_torch,
                    earth2grid.healpix.PixelOrder.NEST,
                    earth2grid.healpix.HEALPIX_PAD_XY
                )
                # Store as (1, 1, npix) to match target shape (B, C, T, X)
                self._land_mask_tensor = land_mask_torch.unsqueeze(0).unsqueeze(0)  # (1, 1, npix)
            else:
                self._land_mask_tensor = None
        
        # Mask loss for TOS channel (channel 1): only compute loss over ocean pixels
        if self._land_mask_tensor is not None and loss.total.shape[1] >= 2:
            # Move land mask to same device as loss
            land_mask = self._land_mask_tensor.to(loss.total.device)  # (1, 1, npix)
            ocean_mask = 1.0 - land_mask  # 1 for ocean, 0 for land
            
            # loss shapes: (B, C, T, X) where C=2 (TAS=0, TOS=1)
            # For TOS channel (index 1), zero out loss over land pixels
            loss_tas = loss.total[:, 0:1, :, :]  # Keep TAS loss as is
            loss_tos = loss.total[:, 1:2, :, :] * ocean_mask  # Zero TOS loss over land
            loss.total = torch.cat([loss_tas, loss_tos], dim=1)
            
            # Also mask denoising loss for consistency
            if loss.denoising.shape[1] >= 2:
                denoising_tas = loss.denoising[:, 0:1, :, :]
                denoising_tos = loss.denoising[:, 1:2, :, :] * ocean_mask
                loss.denoising = torch.cat([denoising_tas, denoising_tos], dim=1)
        
        training_stats.report("Loss/denoising", loss.denoising)
        self._train_sigma_metric.update(loss.sigma, loss.denoising)
        if loss.classification is not None:
            training_stats.report("Loss/classification", loss.classification)
            self._train_classifier_sigma_metric.update(loss.sigma, loss.classification)
        return loss.total

    def _report_log_likelihood(self, net, batch):
        """Optionally skip the extremely slow SciPy log-likelihood integration."""
        if self.skip_log_likelihood:
            return
        return super()._report_log_likelihood(net, batch)
    
    def validate(self, num_batches: int = 50) -> float | None:
        """Compute validation loss on the test set.
        
        Override parent's validate() to:
        1. Skip TensorBoard figure logging (causes size mismatch errors with HEALPix)
        2. Compute and return validation loss for logging
        
        Args:
            num_batches: Number of test batches to evaluate (default: 50)
            
        Returns:
            Average validation loss, or None if validation failed
        """
        import matplotlib.pyplot as plt
        from cbottle import training_stats
        from cbottle import distributed as dist
        
        try:
            # Use the validation loader created during setup (stored as self.valid_loader)
            if not hasattr(self, 'valid_loader') or self.valid_loader is None:
                if dist.get_rank() == 0:
                    print("  -> Validation skipped: no valid_loader available")
                return None
            
            net = self.ema if self.ema is not None else self.net
            
            # Run test batches to compute validation loss
            test_losses = []
            
            test_iter = iter(self.valid_loader)
            for _ in range(num_batches):
                try:
                    batch = next(test_iter)
                except StopIteration:
                    break
                    
                batch = self._stage_dict_batch(batch)
                with torch.no_grad():
                    with torch.autocast("cuda", enabled=self.bf16, dtype=torch.bfloat16):
                        loss = self.test_step(**batch)
                        test_losses.append(loss.mean().item())
            
            if test_losses:
                avg_val_loss = sum(test_losses) / len(test_losses)
                training_stats.report("Loss/val_loss", avg_val_loss)
                
                # Store for access by training loop
                self._last_val_loss = avg_val_loss
                
                if dist.get_rank() == 0:
                    print(f"  -> Validation loss: {avg_val_loss:.4f} (over {len(test_losses)} batches)")
                
                # Close any open figures to prevent memory leaks
                plt.close('all')
                return avg_val_loss
            
            plt.close('all')
            return None
            
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"  -> Validation skipped due to error: {e}")
            plt.close('all')
            return None
    
    def _run_backward_on_batch(self, batch):
        """Run forward and backward pass on a single batch. Returns loss value."""
        from cbottle.training import utils as misc
        
        self.ddp.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        with misc.ddp_sync(self.ddp, True):
            batch = self._stage_dict_batch(batch)
            with torch.autocast("cuda", enabled=self.bf16, dtype=torch.bfloat16):
                loss = self.train_step(**batch)
            loss_value = loss.mean().item()  # Get loss value before backward
            loss.mean().backward()
        
        if self.gradient_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.ddp.parameters(), self.gradient_clip_max_norm
            )
        
        return loss_value
    
    def train(self):
        """
        Training loop with progress bars and comprehensive logging.
        Override parent's train() to add tqdm progress tracking and loss logging.
        """
        from cbottle import distributed as dist
        from cbottle import training_stats
        from cbottle.training import utils as misc
        
        # Setup log file for easy monitoring
        log_file = os.path.join(self.run_dir, "training_log.txt")
        
        def log_msg(msg):
            """Log to both stdout and file."""
            if dist.get_rank() == 0:
                print(msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")
        
        # Initialize
        log_msg("="*60)
        log_msg("MIROC6 Training Started")
        log_msg("="*60)
        log_msg(f"Loss function: {self.loss_fn}")
        log_msg(f"Model output channels: {self.net.img_channels}")
        log_msg(f"Model time_length: {self.net.time_length}")
        log_msg(f"Batch size: {self.batch_size}, per GPU: {self.batch_gpu}")
        log_msg(f"Total ticks: {self.total_ticks}, steps/tick: {self.steps_per_tick}")
        log_msg(f"Learning rate: {self.lr}")
        
        # Check actual first conv layer input channels
        first_conv = None
        for name, module in self.net.named_modules():
            if 'conv' in name.lower() and hasattr(module, 'in_channels'):
                first_conv = module
                break
        if first_conv:
            log_msg(f"First conv in_channels: {first_conv.in_channels}")
        
        log_msg("="*60)
        self._train_curve_data = []
        self._next_plot_step = self.plot_every_steps
        self._next_generate_step = self.generate_every_steps
        
        start_time = time.time()
        np.random.seed(
            (self.seed * dist.get_world_size() + dist.get_rank() + self.cur_nimg)
            % (1 << 31)
        )
        torch.manual_seed(np.random.randint(1 << 31))
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        # Train with progress bars
        tick_start_time = time.time()
        maintenance_time = tick_start_time - start_time
        dist.update_progress(0, self.total_ticks)
        dataset_iterator = iter(self.train_loader)
        
        # Track losses
        tick_losses = []
        all_losses = []
        
        # Outer progress bar for ticks
        tick_pbar = tqdm(
            range(self.total_ticks),
            desc="Training",
            unit="tick",
            disable=(dist.get_rank() != 0),
        )
        
        for cur_tick in tick_pbar:
            tick_losses = []
            
            # Inner progress bar for steps within tick
            step_pbar = tqdm(
                range(self.steps_per_tick),
                desc=f"Tick {cur_tick}",
                unit="step",
                leave=False,
                disable=(dist.get_rank() != 0),
            )
            
            for step_idx in step_pbar:
                # Time data loading
                data_start = time.time()
                batch = next(dataset_iterator)
                data_time = time.time() - data_start
                
                # Time forward/backward and get loss
                compute_start = time.time()
                loss_value = self._run_backward_on_batch(batch)
                compute_time = time.time() - compute_start
                
                tick_losses.append(loss_value)
                all_losses.append(loss_value)
                
                self.cur_nimg += self.batch_size
                self.step_optimizer(self.cur_nimg)
                self.update_ema(self.cur_nimg)
                global_step = self.cur_nimg // self.batch_size
                if dist.get_rank() == 0:
                    self._train_curve_data.append((global_step, loss_value))
                self._maybe_run_aux_hooks(global_step, log_msg)
                
                # Get GPU stats
                gpu_util, mem_used, mem_total = get_gpu_stats()
                
                # Update step progress bar with timing, loss, and GPU info
                postfix = {
                    'loss': f'{loss_value:.4f}',
                    'data': f'{data_time:.2f}s',
                    'fwd': f'{compute_time:.2f}s',
                }
                if gpu_util is not None:
                    postfix['GPU'] = f'{gpu_util}%'
                step_pbar.set_postfix(postfix)
            
            step_pbar.close()
            
            # Calculate tick statistics
            tick_end_time = time.time()
            tick_time = tick_end_time - tick_start_time
            avg_loss = np.mean(tick_losses) if tick_losses else 0
            min_loss = np.min(tick_losses) if tick_losses else 0
            max_loss = np.max(tick_losses) if tick_losses else 0
            
            # Log tick summary
            current_lr = self.optimizer.param_groups[0]['lr']
            log_msg(f"Tick {cur_tick:5d} | loss: {avg_loss:.4f} (min: {min_loss:.4f}, max: {max_loss:.4f}) | "
                   f"lr: {current_lr:.6f} | kimg: {self.cur_nimg/1e3:.1f} | time: {tick_time:.1f}s")
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train_avg', avg_loss, self.cur_nimg)
            self.writer.add_scalar('Loss/train_min', min_loss, self.cur_nimg)
            self.writer.add_scalar('Loss/train_max', max_loss, self.cur_nimg)
            
            self.log_tick(
                maintenance_time,
                tick_start_time,
                tick_end_time,
                start_time,
                cur_tick,
                self.cur_nimg,
            )
            
            # Update outer progress bar
            tick_pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'kimg': f'{self.cur_nimg/1e3:.1f}',
                'sec/tick': f'{tick_time:.1f}',
            })

            # Validate and save checkpoints
            if (
                self.state_dump_ticks is not None
                and (cur_tick + 1) % self.state_dump_ticks == 0
            ):
                self.save_training_state(self.cur_nimg)
                log_msg(f"  -> Saved training state at tick {cur_tick+1}")
            if (
                self.snapshot_ticks is not None
                and (cur_tick + 1) % self.snapshot_ticks == 0
            ):
                self.save_network_snapshot(self.cur_nimg)
                log_msg(f"  -> Saved network snapshot at tick {cur_tick+1}")
                val_loss = self.validate()
                
                # Update progress bar with validation loss
                postfix = {
                    'loss': f'{avg_loss:.4f}',
                    'kimg': f'{self.cur_nimg/1e3:.1f}',
                    'sec/tick': f'{tick_time:.1f}',
                }
                if val_loss is not None:
                    postfix['val_loss'] = f'{val_loss:.4f}'
                tick_pbar.set_postfix(postfix)

            # Update timing
            dist.update_progress(cur_tick + 1, self.total_ticks)
            tick_start_time = time.time()
            maintenance_time = tick_start_time - tick_end_time

        tick_pbar.close()
        self._cleanup_inline_generators(join=True)
        
        # Final summary
        total_time = time.time() - start_time
        final_avg_loss = np.mean(all_losses[-1000:]) if len(all_losses) > 1000 else np.mean(all_losses)
        
        log_msg("="*60)
        log_msg("Training Complete!")
        log_msg("="*60)
        log_msg(f"Total time: {total_time/3600:.2f} hours")
        log_msg(f"Final avg loss (last 1000): {final_avg_loss:.4f}")
        log_msg(f"Total samples: {self.cur_nimg:,}")
        log_msg(f"Total steps: {len(all_losses):,}")
        log_msg("="*60)
        
        # Final save
        self.save_training_state(self.cur_nimg)
        self.save_network_snapshot(self.cur_nimg)
        self.flush_training_stats()

    def _maybe_run_aux_hooks(self, global_step: int, log_fn):
        if dist.get_rank() != 0:
            return

        if self.plot_every_steps:
            if self._next_plot_step is None:
                self._next_plot_step = self.plot_every_steps
            while global_step >= self._next_plot_step:
                self._run_plot_hook(global_step, log_fn)
                self._next_plot_step += self.plot_every_steps

        if self.generate_every_steps:
            if self._next_generate_step is None:
                self._next_generate_step = self.generate_every_steps
            while global_step >= self._next_generate_step:
                self._run_generate_hook(global_step, log_fn)
                self._next_generate_step += self.generate_every_steps

    def _cleanup_inline_generators(self, join: bool = False):
        if not hasattr(self, "_inline_generators"):
            self._inline_generators = []
            return
        if join:
            for thread in self._inline_generators:
                thread.join()
            self._inline_generators = []
            return
        active = []
        for thread in self._inline_generators:
            if thread.is_alive():
                active.append(thread)
        self._inline_generators = active

    def _run_plot_hook(self, step: int, log_fn):
        if not self._train_curve_data:
            return
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Get validation loss if available
        val_loss = getattr(self, '_last_val_loss', None)
        
        # 1. Save to CSV file
        curve_path = os.path.join(self.run_dir, "learning_curve.csv")
        first = not os.path.exists(curve_path)
        
        with open(curve_path, "a", newline="") as f:
            writer = csv.writer(f)
            if first:
                writer.writerow(["step", "train_loss", "val_loss"])
            train_loss = self._train_curve_data[-1][1]
            writer.writerow([step, train_loss, val_loss if val_loss is not None else ""])
        
        # 2. Plot loss curve
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Extract all data points
            steps = [d[0] for d in self._train_curve_data]
            losses = [d[1] for d in self._train_curve_data]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot training loss
            ax.plot(steps, losses, 'b-', linewidth=1.5, alpha=0.7, label='Train Loss')
            
            # Plot validation loss if available
            if val_loss is not None:
                ax.axhline(y=val_loss, color='r', linestyle='--', linewidth=1.5, 
                          alpha=0.7, label=f'Val Loss ({val_loss:.4f})')
            
            # Formatting
            ax.set_xlabel('Step', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f'Training Loss Curve (Step {step})', fontsize=14, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            
            # Add current loss annotation
            current_loss = losses[-1]
            ax.annotate(f'Current: {current_loss:.4f}', 
                       xy=(step, current_loss), 
                       xytext=(10, 10), 
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                       fontsize=10)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.run_dir, "loss_curve.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            log_fn(f"  -> Saved loss curve plot: {plot_path}")
        except Exception as e:
            log_fn(f"  -> Warning: Could not plot loss curve: {e}")
        
        # 3. Log to text file
        log_txt_path = os.path.join(self.run_dir, "training_log.txt")
        with open(log_txt_path, "a") as f:
            val_str = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
            f.write(f"Step {step}: train_loss={train_loss:.6f}{val_str}\n")
        
        val_str = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
        log_fn(f"  -> Logged learning curve entry at step {step}{val_str} -> {curve_path}")

    def _run_generate_hook(self, step: int, log_fn):
        if dist.get_rank() != 0:
            return
        snapshot_dir = os.path.join(self.run_dir, "inline_snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(snapshot_dir, f"step_{step:06d}_nimg_{self.cur_nimg:09d}.pkl")
        manifest = os.path.join(self.run_dir, "generation_requests.txt")
        with open(manifest, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{step},{snapshot_path}\n")
        log_fn(f"  -> Queued inline snapshot for generation at step {step}: {snapshot_path}")
        try:
            self._cleanup_inline_generators()
            if self.max_inline_generators is not None and self.max_inline_generators <= 0:
                self._run_inline_generation(step, snapshot_path, log_fn)
            elif (
                self.max_inline_generators is not None
                and len(self._inline_generators) >= self.max_inline_generators
            ):
                log_fn("  -> Inline generation skipped (generator already running)")
            else:
                thread = threading.Thread(
                    target=self._run_inline_generation,
                    args=(step, snapshot_path, log_fn),
                    daemon=True,
                )
                thread.start()
                self._inline_generators.append(thread)
        except Exception as exc:
            log_fn(f"  -> Inline generation failed at step {step}: {exc}")

    def _run_inline_generation(self, step: int, snapshot_path: str, log_fn):
        """Run sample generation inline during training on rank 0."""
        if not self.generate_every_steps or self.generate_num_samples <= 0:
            log_fn("  -> Inline generation skipped (no samples requested)")
            return

        # Use pre-imported module to avoid import lock deadlock in threads
        gen_module = _gen_module
        if gen_module is None:
            log_fn("  -> Inline generation skipped, generate_miroc6_samples not available")
            return

        device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        base_seed = getattr(self, "seed", 0) or 0
        seed = int(base_seed + step)
        log_fn(f"  -> Running inline generation (samples={self.generate_num_samples}, steps={self.generate_num_steps})")

        generation_start = time.time()
        net = self._materialize_inline_snapshot(snapshot_path, log_fn).to(device).eval()
        with torch.no_grad():
            samples, months = gen_module.generate_samples(
                net,
                num_samples=self.generate_num_samples,
                device=device,
                num_steps=self.generate_num_steps,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                seed=seed,
            )

        samples = gen_module.reorder_to_ring(samples)
        output_root = os.path.join(self.run_dir, "inline_generations")
        output_dir = os.path.join(output_root, f"step_{step:06d}")
        gen_module.plot_samples(samples, months, output_dir)
        elapsed = time.time() - generation_start
        log_fn(f"  -> Inline generation complete in {elapsed:.1f}s -> {output_dir}")
        del net, samples
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _materialize_inline_snapshot(self, snapshot_path: str, log_fn):
        """Create a lightweight snapshot for inline generation and return the EMA network."""
        os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
        with torch.no_grad():
            ema_copy = copy.deepcopy(self.ema).eval().requires_grad_(False).cpu()
        with open(snapshot_path, "wb") as f:
            pickle.dump({"ema": ema_copy}, f)
        log_fn(f"  -> Inline snapshot written to {snapshot_path}")
        return ema_copy


@dataclasses.dataclass
class CLI(training_loop_base.CLI):
    """CLI for MIROC6 unconditional training."""
    
    loop: MirocTrainingLoop = dataclasses.field(
        default_factory=lambda: MirocTrainingLoop(
            # Training schedule
            total_ticks=100_000,
            steps_per_tick=500,
            state_dump_ticks=5,
            snapshot_ticks=5,
            
            # Batch settings
            batch_size=64,
            batch_gpu=4,
            
            # Learning rate
            lr=1e-4,
            lr_rampup_img=10_000,
            lr_flat_imgs=500_000,
            lr_decay_imgs=500_000,
            lr_min=1e-6,
            
            # EDM loss settings (from Karras et al. 2022)
            p_mean=-1.2,
            p_std=1.2,
            sigma_min=0.002,
            sigma_max=1000.0,
            noise_distribution="log_normal",
            
            # Unconditional settings
            with_era5=False,
            use_labels=False,
            monthly_sst_input=False,
            variables="miroc",
            
            # Data loading
            dataloader_num_workers=4,
            dataloader_prefetch_factor=100,
        )
    )


def main():
    cli = parse_args(CLI, convert_underscore_to_hyphen=False)
    logging.basicConfig(level=logging.INFO)
    loop = cli.loop

    if not cli.name:
        name = "miroc6-uncond-" + str(int(time.time()))
    else:
        name = cli.name

    loop.run_dir = os.path.join(cli.output_dir, name)

    dist.init()
    
    # Handle test_fast mode
    if cli.test_fast:
        loop.total_ticks = 2
        loop.steps_per_tick = 1
        loop.state_dump_ticks = 1
        loop.snapshot_ticks = 1
        loop.batch_size = 1 * dist.get_world_size()
        loop.batch_gpu = 1
        loop.lr_rampup_img = 10_000
        loop.valid_min_samples = 2 * dist.get_world_size()

    print("=" * 60)
    print("MIROC6 Unconditional Diffusion Training")
    print("Following Rozet et al. 2023 Score-based Data Assimilation")
    print("=" * 60)
    print(f"Variables: {loop.variable_config.variables_2d}")
    print(f"Time window: {loop.time_window_months} months")
    print(f"Batch size: {loop.batch_size} (per GPU: {loop.batch_gpu})")
    print(f"Model channels: {loop.network.model_channels}")
    print(f"Output channels: {loop.out_channels}")
    print(f"Output dir: {loop.run_dir}")
    
    # Print model config before setup
    mc = loop.model_config
    print(f"Model config: out_channels={mc.out_channels}, condition_channels={mc.condition_channels}, label_dim={mc.label_dim}")
    print(f"Model config total input channels: {mc.out_channels + mc.condition_channels}")
    print("=" * 60)
    
    loop.setup()

    # Resume logic
    resume_dirs_in_priority = [loop.run_dir, cli.resume_dir]
    new_training = True
    for rundir in resume_dirs_in_priority:
        try:
            loop.resume_from_rundir(rundir, require_all=False)
            new_training = False
            print(f"Resumed from {rundir}")
            break
        except FileNotFoundError:
            pass
        except (TypeError, ValueError) as e:
            # Happens when run_dir exists but has no valid training state
            # (e.g., from a previous failed/incomplete run)
            print(f"Warning: Could not resume from {rundir}: {e}")
            pass

    if new_training:
        print("Starting new training")

    if cli.validate_only:
        loop.validate()
        loop.flush_training_stats()
    elif cli.validate_all:
        loop.validate_all_checkpoints()
    else:
        loop.train()


if __name__ == "__main__":
    main()
