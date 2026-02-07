"""
UniMol Weight Manager

Automatically manages UniMol model weights with support for:
- Local weight detection
- Automatic download from HuggingFace
- Caching for offline use
"""

import os
import sys
import torch
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class UniMolWeightManager:
    """Manages UniMol model weights with automatic download support."""

    # HuggingFace model URLs
    HUGGINGFACE_MODELS = {
        'unimolv1': {
            'molecule_all_h': {
                'repo': 'dptech/Uni-Mol-Models',
                'file': 'mol_pre_all_h_220816.pt',
                'dict': 'mol.dict.txt'
            },
            'oled_no_h': {
                'repo': 'dptech/Uni-Mol-Models',
                'file': 'oled_pre_no_h_230101.pt',
                'dict': 'oled.dict.txt'
            }
        },
        'unimolv2': {
            '84m': {
                'repo': 'dptech/Uni-Mol2-Models',
                'file': 'checkpoint_84m.pt',
                'config': 'config_84m.json'
            },
            '164m': {
                'repo': 'dptech/Uni-Mol2-Models',
                'file': 'checkpoint_164m.pt',
                'config': 'config_164m.json'
            },
            '310m': {
                'repo': 'dptech/Uni-Mol2-Models',
                'file': 'checkpoint_310m.pt',
                'config': 'config_310m.json'
            },
            '570m': {
                'repo': 'dptech/Uni-Mol2-Models',
                'file': 'checkpoint_570m.pt',
                'config': 'config_570m.json'
            },
            '1.1B': {
                'repo': 'dptech/Uni-Mol2-Models',
                'file': 'checkpoint_1.1B.pt',
                'config': 'config_1.1B.json'
            }
        }
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize weight manager.

        Args:
            cache_dir: Directory to cache downloaded weights.
                      Defaults to ~/.cache/unimol_weights
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser('~/.cache/unimol_weights')

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Weight cache directory: {self.cache_dir}")

    def get_weight_path(
        self,
        model_version: str,
        model_type: str,
        local_path: Optional[str] = None,
        force_download: bool = False
    ) -> Dict[str, str]:
        """
        Get path to model weights, downloading if necessary.

        Args:
            model_version: 'unimolv1' or 'unimolv2'
            model_type: Specific model type (e.g., 'molecule_all_h', '84m', '1.1B')
            local_path: Optional path to local weights (takes priority)
            force_download: Force re-download even if cached

        Returns:
            Dictionary with paths to weights and config files
        """
        # Method 1: Use local path if provided
        if local_path and os.path.exists(local_path):
            logger.info(f"✓ Using local weights: {local_path}")
            return self._get_local_weights(local_path, model_version)

        # Method 2: Check cache
        if not force_download:
            cached = self._check_cache(model_version, model_type)
            if cached:
                logger.info(f"✓ Using cached weights")
                return cached

        # Method 3: Download from HuggingFace
        logger.info(f"⬇️  Downloading weights from HuggingFace...")
        return self._download_weights(model_version, model_type)

    def _get_local_weights(self, local_path: str, model_version: str) -> Dict[str, str]:
        """Get weights from local path."""
        weight_path = Path(local_path)

        if not weight_path.exists():
            raise FileNotFoundError(f"Local weights not found: {local_path}")

        result = {
            'weights': str(weight_path),
            'source': 'local'
        }

        # Look for associated files
        if model_version == 'unimolv1':
            # Look for dictionary file
            dict_path = weight_path.parent / 'mol.dict.txt'
            if not dict_path.exists():
                dict_path = weight_path.parent / 'oled.dict.txt'
            if dict_path.exists():
                result['dict'] = str(dict_path)

        elif model_version == 'unimolv2':
            # Look for config file
            config_path = weight_path.parent / 'config.json'
            if config_path.exists():
                result['config'] = str(config_path)

        return result

    def _check_cache(self, model_version: str, model_type: str) -> Optional[Dict[str, str]]:
        """Check if weights are cached."""
        cache_subdir = self.cache_dir / model_version / model_type

        if not cache_subdir.exists():
            return None

        result = {'source': 'cache'}

        if model_version == 'unimolv1':
            weight_file = cache_subdir / self.HUGGINGFACE_MODELS[model_version][model_type]['file']
            dict_file = cache_subdir / self.HUGGINGFACE_MODELS[model_version][model_type]['dict']

            if weight_file.exists() and dict_file.exists():
                result['weights'] = str(weight_file)
                result['dict'] = str(dict_file)
                return result

        elif model_version == 'unimolv2':
            weight_file = cache_subdir / self.HUGGINGFACE_MODELS[model_version][model_type]['file']
            config_file = cache_subdir / self.HUGGINGFACE_MODELS[model_version][model_type]['config']

            if weight_file.exists() and config_file.exists():
                result['weights'] = str(weight_file)
                result['config'] = str(config_file)
                return result

        return None

    def _download_weights(self, model_version: str, model_type: str) -> Dict[str, str]:
        """Download weights from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface-hub is required for automatic weight download. "
                "Install it with: pip install huggingface-hub"
            )

        if model_version not in self.HUGGINGFACE_MODELS:
            raise ValueError(f"Unknown model version: {model_version}")

        if model_type not in self.HUGGINGFACE_MODELS[model_version]:
            raise ValueError(f"Unknown model type: {model_type}")

        model_info = self.HUGGINGFACE_MODELS[model_version][model_type]
        cache_subdir = self.cache_dir / model_version / model_type
        cache_subdir.mkdir(parents=True, exist_ok=True)

        result = {'source': 'downloaded'}

        # Download weight file
        weight_path = hf_hub_download(
            repo_id=model_info['repo'],
            filename=model_info['file'],
            cache_dir=str(self.cache_dir),
            local_dir=str(cache_subdir),
            local_dir_use_symlinks=False
        )
        result['weights'] = weight_path
        logger.info(f"  ✓ Downloaded: {model_info['file']}")

        # Download additional files
        if 'dict' in model_info:
            dict_path = hf_hub_download(
                repo_id=model_info['repo'],
                filename=model_info['dict'],
                cache_dir=str(self.cache_dir),
                local_dir=str(cache_subdir),
                local_dir_use_symlinks=False
            )
            result['dict'] = dict_path
            logger.info(f"  ✓ Downloaded: {model_info['dict']}")

        if 'config' in model_info:
            config_path = hf_hub_download(
                repo_id=model_info['repo'],
                filename=model_info['config'],
                cache_dir=str(self.cache_dir),
                local_dir=str(cache_subdir),
                local_dir_use_symlinks=False
            )
            result['config'] = config_path
            logger.info(f"  ✓ Downloaded: {model_info['config']}")

        return result

    def list_available_models(self) -> Dict[str, Dict]:
        """List all available models."""
        return self.HUGGINGFACE_MODELS

    def clear_cache(self, model_version: Optional[str] = None):
        """Clear cached weights."""
        if model_version:
            cache_path = self.cache_dir / model_version
        else:
            cache_path = self.cache_dir

        if cache_path.exists():
            import shutil
            shutil.rmtree(cache_path)
            logger.info(f"✓ Cleared cache: {cache_path}")


# Convenience functions
def get_weights(
    model_version: str = 'unimolv1',
    model_type: str = 'molecule_all_h',
    local_path: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Convenience function to get model weights.

    Args:
        model_version: 'unimolv1' or 'unimolv2'
        model_type: Specific model type
        local_path: Optional path to local weights
        cache_dir: Optional cache directory

    Returns:
        Dictionary with paths to weights and config files
    """
    manager = UniMolWeightManager(cache_dir=cache_dir)
    return manager.get_weight_path(model_version, model_type, local_path)


if __name__ == '__main__':
    # Test weight manager
    logging.basicConfig(level=logging.INFO)

    manager = UniMolWeightManager()

    print("\nAvailable models:")
    for version, models in manager.list_available_models().items():
        print(f"\n{version}:")
        for model_type in models.keys():
            print(f"  - {model_type}")

    print("\n" + "="*70)
    print("Testing weight download (UniMol v1, molecule_all_h)...")
    print("="*70)

    try:
        paths = manager.get_weight_path('unimolv1', 'molecule_all_h')
        print("\n✓ Success!")
        print(f"  Weights: {paths['weights']}")
        if 'dict' in paths:
            print(f"  Dict: {paths['dict']}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
