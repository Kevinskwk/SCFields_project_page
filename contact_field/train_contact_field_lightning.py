#!/usr/bin/env python3
"""
PyTorch Lightning training script for contact field estimation
"""

import os
import sys
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from datetime import datetime

# Set multiprocessing start method to 'spawn' to avoid CUDA issues
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models import create_model, get_model_info
from models.module import ContactFieldTrainingModule
from dataset import ContactFieldDataset


def create_callbacks(cfg: DictConfig, checkpoint_dir: Path) -> list:
    """Create PyTorch Lightning callbacks"""
    callbacks = []

    # Best model checkpointing (save top-k best models based on validation loss)
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best-contact_field-{epoch:02d}-{step:06d}',
        monitor='val/total_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
        verbose=True
    )
    callbacks.append(best_checkpoint_callback)

    # Regular periodic checkpointing (save every N epochs regardless of performance)
    save_frequency = cfg.training.get('save_frequency', 10)
    if save_frequency > 0:
        periodic_checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir / 'periodic',
            filename='periodic-contact_field-{epoch:02d}',
            every_n_epochs=save_frequency,
            save_top_k=-1,  # Save all periodic checkpoints
            verbose=True
        )
        callbacks.append(periodic_checkpoint_callback)

    # Early stopping
    if cfg.training.get('early_stopping', {}).get('enabled', False):
        early_stop_callback = EarlyStopping(
            monitor='val/total_loss',
            patience=cfg.training.early_stopping.get('patience', 10),
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop_callback)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    return callbacks


def create_logger(cfg: DictConfig, checkpoint_dir: Path) -> WandbLogger:
    """Create Weights & Biases logger"""
    # Get wandb config from main config or use defaults
    wandb_config = cfg.get('wandb', {})

    # Extract project and entity with defaults
    project_name = wandb_config.get('project', 'contact-field-training')
    entity = wandb_config.get('entity', None)

    # Create experiment name
    experiment_name = cfg.logging.get('experiment_name', 'contact_field_experiment')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment_name}_{timestamp}"

    # Create WandB logger
    logger = WandbLogger(
        project=project_name,
        entity=entity,
        name=run_name,
        save_dir=str(checkpoint_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[
            cfg.model.type,
            f"loss_{cfg.training.loss.contact_prob_loss_type}",
            cfg.data.task
        ],
        offline=wandb_config.get('offline', False)
    )

    return logger


def setup_directories(cfg: DictConfig) -> Path:
    """Setup experiment directories"""
    # Setup experiment directory
    base_checkpoint_dir = Path(cfg.checkpoint.save_dir)

    if cfg.checkpoint.get('use_timestamp', False):
        experiment_name = cfg.logging.get('experiment_name', 'contact_field_experiment')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = base_checkpoint_dir / f"{experiment_name}_{timestamp}"
    else:
        checkpoint_dir = base_checkpoint_dir

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create periodic checkpoint subdirectory
    periodic_dir = checkpoint_dir / 'periodic'
    periodic_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment directory: {checkpoint_dir}")
    print(f"Best model checkpoints will be saved to: {checkpoint_dir}")
    print(f"Periodic checkpoints will be saved to: {periodic_dir}")

    # Save config to experiment directory
    config_path = checkpoint_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f)
    print(f"Config saved to: {config_path}")

    return checkpoint_dir


@hydra.main(version_base=None, config_path="config", config_name="peg_config")
def main(cfg: DictConfig) -> None:
    """Main training function"""
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Seed all RNGs for reproducibility.
    seed = int(cfg.get('seed', 42))
    pl.seed_everything(seed, workers=True)
    print(f"Random seed: {seed}")

    # Setup directories
    checkpoint_dir = setup_directories(cfg)

    # Setup device and show GPU information
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")

    # Show which devices will be used based on config
    accelerator = cfg.training.get('accelerator', 'auto')
    devices = cfg.training.get('devices', 'auto')
    strategy = cfg.training.get('strategy', 'auto')
    print(f"\nTraining configuration:")
    print(f"  Accelerator: {accelerator}")
    print(f"  Devices: {devices}")
    print(f"  Strategy: {strategy}")
    print(f"  Precision: {cfg.training.get('precision', 32)}")

    # Create model
    model = create_model(cfg)

    # Print model information
    model_info = get_model_info(cfg)
    print(f"Model: {model_info['description']}")
    print(f"Class: {model_info['class_name']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Uses environment: {model_info['uses_environment']}")
    print(f"Architecture: {model_info['architecture']}")

    # Create train/validation data objects
    train_dataset = ContactFieldDataset(
        cfg.data.train_data_path,
        cfg.data,
        split='train',
        real_data=cfg.data.get('real_data', False),
    )
    val_dataset = ContactFieldDataset(
        cfg.data.val_data_path,
        cfg.data,
        split='val',
        real_data=cfg.data.get('real_data', False),
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.get('num_workers', 31)),
        pin_memory=cfg.get('pin_memory', True),
        persistent_workers=True if int(cfg.get('num_workers', 31)) > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.get('num_workers', 31)),
        pin_memory=cfg.get('pin_memory', True),
        persistent_workers=True if int(cfg.get('num_workers', 31)) > 0 else False
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # Print dataset information
    train_cache_info = train_dataset.get_cache_info()
    val_cache_info = val_dataset.get_cache_info()
    print(f"\nTrain dataset info:")
    print(f"  Total samples: {train_cache_info['total_samples']}")
    print(f"  Loaded samples: {train_cache_info['loaded_samples']}")
    print(f"  Data preloaded: {train_cache_info['data_preloaded']}")
    print(f"  Storage device: {train_cache_info['device']}")
    print(f"  Estimated memory usage: {train_cache_info['estimated_memory_gb']:.2f} GB")
    print(f"Val dataset info:")
    print(f"  Total samples: {val_cache_info['total_samples']}")
    print(f"  Loaded samples: {val_cache_info['loaded_samples']}")
    print(f"  Data preloaded: {val_cache_info['data_preloaded']}")
    print(f"  Storage device: {val_cache_info['device']}")
    print(f"  Estimated memory usage: {val_cache_info['estimated_memory_gb']:.2f} GB")

    # Create PyTorch Lightning module
    pl_module = ContactFieldTrainingModule(cfg, model)

    # Load initial weights if specified (for fine-tuning)
    # This loads ONLY model weights, not optimizer state or epoch counter
    init_weights_path = cfg.checkpoint.get('load_weights_from', None)
    if init_weights_path and os.path.exists(init_weights_path):
        print(f"\n{'='*80}")
        print(f"Loading initial weights from: {init_weights_path}")
        print(f"Mode: Fine-tuning (weights only, no optimizer/epoch state)")
        print(f"{'='*80}\n")

        # Load checkpoint
        checkpoint = torch.load(init_weights_path, map_location='cpu', weights_only=False)

        # Extract model state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'network.' prefix if present (Lightning adds this)
            if any(k.startswith('network.') for k in state_dict.keys()):
                state_dict = {k.replace('network.', '', 1): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint

        # Load weights (strict=False allows partial loading if model changed)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"⚠️  Missing keys (new parameters, will be randomly initialized):")
            for key in missing_keys[:10]:  # Show first 10
                print(f"   - {key}")
            if len(missing_keys) > 10:
                print(f"   ... and {len(missing_keys)-10} more")

        if unexpected_keys:
            print(f"⚠️  Unexpected keys (in checkpoint but not in model):")
            for key in unexpected_keys[:10]:
                print(f"   - {key}")
            if len(unexpected_keys) > 10:
                print(f"   ... and {len(unexpected_keys)-10} more")

        if not missing_keys and not unexpected_keys:
            print("✅ All weights loaded successfully!")

        print(f"\nInitialization checkpoint info:")
        if 'epoch' in checkpoint:
            print(f"  Original epoch: {checkpoint['epoch']}")
        if 'global_step' in checkpoint:
            print(f"  Original global_step: {checkpoint['global_step']}")
        print(f"  Starting fresh training from epoch 0\n")

    # Create logger
    logger = create_logger(cfg, checkpoint_dir)

    # Count parameters after potential freezing
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Print parameter summary
    print(f"\nParameter Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"  Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")

    # Log additional experiment information
    logger.log_hyperparams({
        'model/total_parameters': total_params,
        'model/trainable_parameters': trainable_params,
        'model/frozen_parameters': frozen_params,
        'model/trainable_ratio': trainable_params / total_params,
        'data/train_samples': len(train_dataset),
        'data/val_samples': len(val_dataset),
        'data/train_memory_gb': train_cache_info['estimated_memory_gb'],
        'data/val_memory_gb': val_cache_info['estimated_memory_gb'],
        'data/train_preloaded': train_cache_info['data_preloaded'],
        'data/val_preloaded': val_cache_info['data_preloaded'],
    })

    # Log initialization checkpoint if used
    if init_weights_path and os.path.exists(init_weights_path):
        logger.log_hyperparams({
            'initial_weights/checkpoint_path': init_weights_path,
            'initial_weights/weights_only': True,
        })

    # Create callbacks
    callbacks = create_callbacks(cfg, checkpoint_dir)

    # Create trainer
    # GPU Configuration Examples:
    # 1. Single GPU:
    #    accelerator='gpu', devices=1 or devices=[0]
    # 2. Multiple specific GPUs:
    #    accelerator='gpu', devices=[0,1,2] or devices="0,1,2"
    # 3. All available GPUs:
    #    accelerator='gpu', devices=-1 or devices='auto'
    # 4. Auto-detect best setup:
    #    accelerator='auto', devices='auto'
    # 5. CPU only:
    #    accelerator='cpu', devices=1
    #
    # Multi-GPU Strategies:
    # - 'ddp': Distributed Data Parallel (recommended for multi-node/multi-GPU)
    # - 'ddp_spawn': DDP with spawn (slower but more stable)
    # - 'dp': DataParallel (single-node only, not recommended)
    # - 'fsdp': Fully Sharded Data Parallel (for very large models)
    trainer = pl.Trainer(
        max_epochs=int(cfg.training.num_epochs),
        accelerator=cfg.training.get('accelerator', 'gpu'),  # 'gpu', 'cpu', or 'auto'
        devices=cfg.training.get('devices', 'auto'),           # GPU specification
        strategy=cfg.training.get('strategy', 'auto'),         # Multi-GPU strategy
        precision=cfg.training.get('precision', 32),
        gradient_clip_val=cfg.training.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=cfg.training.get('accumulate_grad_batches', 1),
        check_val_every_n_epoch=int(cfg.training.get('val_frequency', 1)),
        log_every_n_steps=int(cfg.logging.get('log_frequency', 50)),
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=cfg.get('deterministic', False),
        benchmark=cfg.get('benchmark', True),
        profiler=cfg.get('profiler', None),
    )

    # Determine checkpoint loading strategy
    # Priority: load_weights_from (fine-tuning) > resume_from (full resume)
    init_weights_loaded = cfg.checkpoint.get('load_weights_from', None) and os.path.exists(cfg.checkpoint.get('load_weights_from', ''))

    resume_path = None
    if not init_weights_loaded:  # Only use resume_from if we didn't load initial weights
        resume_path = cfg.checkpoint.get('resume_from', None) or cfg.get('resume_from', None)
        if resume_path and os.path.exists(resume_path):
            print(f"\n{'='*80}")
            print(f"Resuming full training state from: {resume_path}")
            print(f"Mode: Resume (weights + optimizer + epoch state)")
            print(f"{'='*80}\n")
        else:
            resume_path = None

    # Start training
    print("Starting training...")
    trainer.fit(
        model=pl_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_path  # None for fresh start or fine-tuning, path for resume
    )

    print("Training completed!")

    # Test the best model on validation set
    # best_checkpoint_callback = None
    # for callback in callbacks:
    #     if isinstance(callback, ModelCheckpoint) and callback.monitor == 'val/total_loss':
    #         best_checkpoint_callback = callback
    #         break

    # if best_checkpoint_callback and best_checkpoint_callback.best_model_path:
    #     print(f"Testing best model: {best_checkpoint_callback.best_model_path}")
    #     trainer.test(
    #         model=pl_module,
    #         dataloaders=val_loader,
    #         ckpt_path=best_checkpoint_callback.best_model_path
    #     )

    # Finalize wandb
    if logger:
        logger.finalize("success")


if __name__ == "__main__":
    main()
