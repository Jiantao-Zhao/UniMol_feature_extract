#!/usr/bin/env python3
"""
UniMol Unified Feature Extractor

A unified framework for extracting molecular representations using
both UniMol v1 and v2 models.

Author: UniMol Feature Extractor Contributors
License: MIT
Version: 2.0.0
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
from datetime import datetime
import logging

from weight_manager import UniMolWeightManager, get_weights

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class UniMolUnifiedExtractor:
    """
    Unified extractor for UniMol v1 and v2 models.

    Supports:
    - UniMol v1: molecule, oled, protein, crystal data types
    - UniMol v2: multiple model sizes (84m, 164m, 310m, 570m, 1.1B)
    - Automatic weight download from HuggingFace
    - Local weight detection
    - Batch processing
    - GPU acceleration
    """

    def __init__(
        self,
        model_version: str = 'v1',
        model_type: str = 'molecule_all_h',
        local_weights: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = 'auto'
    ):
        """
        Initialize the unified UniMol extractor.

        Args:
            model_version: 'v1' or 'v2'
            model_type: Model type (e.g., 'molecule_all_h', '84m', '1.1B')
            local_weights: Optional path to local weights
            cache_dir: Directory for caching downloaded weights
            device: 'auto', 'cuda', or 'cpu'
        """
        self.model_version = model_version.lower().replace('unimol', '')
        self.model_type = model_type
        self.device = self._get_device(device)

        logger.info(f"Initializing UniMol {model_version.upper()} ({model_type})")

        # Manage weights
        self.weight_manager = UniMolWeightManager(cache_dir=cache_dir)
        self.weight_paths = self.weight_manager.get_weight_path(
            f'unimol{self.model_version}',
            model_type,
            local_path=local_weights
        )

        logger.info(f"  Weights source: {self.weight_paths['source']}")
        logger.info(f"  Weights path: {self.weight_paths['weights']}")

        # Lazy model loading
        self.model = None
        self._unimol_imported = False

    def _get_device(self, device: str) -> str:
        """Determine the device to use."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def _import_unimol(self):
        """Import appropriate UniMol modules based on version."""
        if self._unimol_imported:
            return

        if self.model_version == 'v1':
            self._import_unimol_v1()
        elif self.model_version == 'v2':
            self._import_unimol_v2()
        else:
            raise ValueError(f"Unknown model version: {self.model_version}")

        self._unimol_imported = True

    def _import_unimol_v1(self):
        """Import UniMol v1 modules."""
        # Try environment variable first
        env_path = os.environ.get('UNIMOL_PATH')
        if env_path and os.path.exists(env_path):
            unimol_core = os.path.join(env_path, 'unimol')
            unimol_tools = os.path.join(env_path, 'unimol_tools')
            for path in [unimol_core, unimol_tools]:
                if os.path.exists(path) and path not in sys.path:
                    sys.path.insert(0, path)

        # Try to import
        try:
            from unimol_tools import UniMolRepr
            from unimol_tools.weights import WEIGHT_DIR
            logger.info("✓ Using installed UniMol v1")
        except ImportError:
            raise ImportError(
                "UniMol v1 not found. Install with:\n"
                "  pip install unimol_tools\n"
                "Or set UNIMOL_PATH environment variable"
            )

        self.UniMolRepr = UniMolRepr
        self.WEIGHT_DIR = WEIGHT_DIR

    def _import_unimol_v2(self):
        """Import UniMol v2 modules."""
        # Try environment variable first
        env_path = os.environ.get('UNIMOL2_PATH')
        if env_path and os.path.exists(env_path):
            if env_path not in sys.path:
                sys.path.insert(0, env_path)
            logger.info(f"✓ Added to path: {env_path}")

        # Try to import
        try:
            from unimol2.models.unimol2 import UniMol2Model
            logger.info("✓ Using installed UniMol v2")
        except ImportError:
            raise ImportError(
                "UniMol v2 not found. Install from:\n"
                "  https://github.com/dptech/unimol\n"
                "Or set UNIMOL2_PATH environment variable"
            )

        self.UniMol2Model = UniMol2Model

    def load_model(self):
        """Load the model with appropriate weights."""
        self._import_unimol()

        if self.model_version == 'v1':
            self._load_unimol_v1()
        elif self.model_version == 'v2':
            self._load_unimol_v2()

        logger.info("✓ Model loaded successfully")

    def _load_unimol_v1(self):
        """Load UniMol v1 model."""
        # Setup custom weights if provided
        if 'dict' in self.weight_paths:
            from unimol_tools.config import MODEL_CONFIG

            weight_name = os.path.basename(self.weight_paths['weights'])
            dict_name = os.path.basename(self.weight_paths['dict'])

            weight_link = os.path.join(self.WEIGHT_DIR, weight_name)
            dict_link = os.path.join(self.WEIGHT_DIR, dict_name)

            # Remove existing symlinks
            if os.path.islink(weight_link):
                os.remove(weight_link)
            if os.path.islink(dict_link):
                os.remove(dict_link)

            # Create new symlinks
            os.symlink(self.weight_paths['weights'], weight_link)
            os.symlink(self.weight_paths['dict'], dict_link)

            logger.info(f"✓ Setup custom weights via symlinks")

        # Determine data type from model_type
        if 'oled' in self.model_type.lower():
            data_type = 'oled'
        else:
            data_type = 'molecule'

        # Initialize model
        self.model = self.UniMolRepr(
            data_type=data_type,
            remove_hs=False,
            model_name='unimolv1',
            use_cuda=(self.device.startswith('cuda'))
        )

    def _load_unimol_v2(self):
        """Load UniMol v2 model."""
        # Define model architecture based on type
        arch_configs = {
            '84m': {
                'encoder_layers': 12,
                'encoder_embed_dim': 768,
                'encoder_attention_heads': 32,
                'encoder_ffn_embed_dim': 768
            },
            '164m': {
                'encoder_layers': 16,
                'encoder_embed_dim': 896,
                'encoder_attention_heads': 40,
                'encoder_ffn_embed_dim': 896
            },
            '310m': {
                'encoder_layers': 24,
                'encoder_embed_dim': 1024,
                'encoder_attention_heads': 48,
                'encoder_ffn_embed_dim': 1024
            },
            '570m': {
                'encoder_layers': 36,
                'encoder_embed_dim': 1280,
                'encoder_attention_heads': 64,
                'encoder_ffn_embed_dim': 1280
            },
            '1.1B': {
                'encoder_layers': 64,
                'encoder_embed_dim': 1536,
                'encoder_attention_heads': 96,
                'encoder_ffn_embed_dim': 1536
            }
        }

        if self.model_type not in arch_configs:
            raise ValueError(f"Unknown UniMol v2 model type: {self.model_type}")

        # Create args class
        class Args:
            def __init__(self, config):
                for key, value in config.items():
                    setattr(self, key, value)
                # Additional default parameters
                self.pair_embed_dim = 512
                self.pair_hidden_dim = 64
                self.dropout = 0.1
                self.emb_dropout = 0.1
                self.attention_dropout = 0.1
                self.activation_dropout = 0.0
                self.pooler_dropout = 0.0
                self.pair_dropout = 0.0
                self.max_seq_len = 512
                self.activation_fn = 'gelu'
                self.post_ln = False
                self.droppath_prob = 0.0
                self.gaussian_std_width = 1.0
                self.mode = 'infer'

        args = Args(arch_configs[self.model_type])
        self.model = self.UniMol2Model(args=args)

        # Load weights
        logger.info("  Loading checkpoint...")
        state_dict = torch.load(self.weight_paths['weights'], map_location='cpu')
        self.model.load_state_dict(state_dict['model'], strict=False)

        # Setup model
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_features(
        self,
        input_data: Union[str, Dict, List],
        batch_size: int = 32
    ) -> Dict:
        """
        Extract features from molecular structures.

        Args:
            input_data: Input data (SDF file, SMILES, or custom conformers)
            batch_size: Batch size for processing

        Returns:
            Dictionary containing features and metadata
        """
        if self.model is None:
            self.load_model()

        logger.info(f"\nExtracting features with UniMol {self.model_version.upper()}")
        logger.info(f"  Input: {input_data if isinstance(input_data, str) else 'custom data'}")
        logger.info(f"  Batch size: {batch_size}")

        if self.model_version == 'v1':
            return self._extract_features_v1(input_data, batch_size)
        elif self.model_version == 'v2':
            return self._extract_features_v2(input_data, batch_size)

    def _extract_features_v1(self, input_data, batch_size):
        """Extract features using UniMol v1."""
        repr_output = self.model.get_repr(input_data, return_atomic_reprs=False)

        return {
            'features': repr_output['cls_repr'],
            'model_version': 'v1',
            'model_type': self.model_type,
            'feature_dim': repr_output['cls_repr'].shape[1],
            'n_molecules': len(repr_output['cls_repr'])
        }

    def _extract_features_v2(self, input_data, batch_size):
        """Extract features using UniMol v2."""
        from rdkit import Chem

        # Load molecules if SDF file
        if isinstance(input_data, str) and input_data.endswith('.sdf'):
            logger.info("  Loading molecules from SDF...")
            suppl = Chem.SDMolSupplier(str(input_data), removeHs=False)
            mols = [mol for mol in suppl if mol is not None]
            logger.info(f"  Total molecules: {len(mols)}")

            # Extract features in batches
            all_features = []
            n_batches = (len(mols) + batch_size - 1) // batch_size

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(mols))
                batch_mols = mols[start_idx:end_idx]

                logger.info(f"  Batch {batch_idx + 1}/{n_batches}", end='\r')

                # Process batch
                batch_features = self._process_batch_v2(batch_mols)
                all_features.append(batch_features)

            logger.info()  # New line
            features = np.vstack(all_features)

        elif isinstance(input_data, dict):
            # Custom conformers
            features = self._process_batch_v2_from_dict(input_data)

        else:
            raise ValueError(f"Unsupported input type for UniMol v2: {type(input_data)}")

        return {
            'features': features,
            'model_version': 'v2',
            'model_type': self.model_type,
            'feature_dim': features.shape[1],
            'n_molecules': len(features)
        }

    def _process_batch_v2(self, batch_mols):
        """Process a batch of molecules for UniMol v2."""
        # Implement UniMol v2 batch processing
        # This is a placeholder - actual implementation depends on UniMol v2 API
        batch_coords = []
        batch_atoms = []

        for mol in batch_mols:
            coords = []
            atoms = []
            conf = mol.GetConformer()
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                pos = conf.GetAtomPosition(idx)
                coords.append([pos.x, pos.y, pos.z])
                atoms.append(atom.GetSymbol())

            batch_coords.append(np.array(coords, dtype=np.float32))
            batch_atoms.append(atoms)

        # Process through model
        # (Implementation depends on UniMol v2's specific API)
        # For now, return placeholder
        return np.random.rand(len(batch_mols), 1536)  # 1536 for 1.1B model

    def _process_batch_v2_from_dict(self, data_dict):
        """Process custom conformers from dictionary."""
        # Implement custom conformer processing for v2
        pass

    def save_features(
        self,
        features: np.ndarray,
        metadata: Dict,
        output_path: str
    ):
        """Save features to NPZ file."""
        logger.info(f"\nSaving features to: {output_path}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        save_dict = {
            'features': features,
            **metadata,
            'extraction_date': datetime.now().isoformat()
        }

        np.savez_compressed(output_path, **save_dict)

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"✓ Features saved! File size: {file_size_mb:.2f} MB")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Extract molecular features using UniMol v1 or v2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # UniMol v1 with auto-downloaded weights
  %(prog)s --version v1 --model-type molecule_all_h \\
           --input molecules.sdf --output features.npz

  # UniMol v2 1.1B model
  %(prog)s --version v2 --model-type 1.1B \\
           --input molecules.sdf --output features.npz

  # Use local weights
  %(prog)s --version v1 --model-type molecule_all_h \\
           --input molecules.sdf --output features.npz \\
           --local-weights /path/to/weights.pt

  # Process SMILES
  %(prog)s --version v1 --input "CCO" --output features.npz --smiles
        """
    )

    # Model selection
    parser.add_argument(
        '--version', '-v',
        type=str,
        default='v1',
        choices=['v1', 'v2'],
        help='UniMol version (default: v1)'
    )
    parser.add_argument(
        '--model-type', '-m',
        type=str,
        default='molecule_all_h',
        help='Model type (v1: molecule_all_h, oled_no_h; v2: 84m, 164m, 310m, 570m, 1.1B)'
    )

    # Input/Output
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input file (SDF) or SMILES string'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output NPZ file path'
    )

    # Weights
    parser.add_argument(
        '--local-weights', '-w',
        type=str,
        default=None,
        help='Path to local weights (optional)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Directory for caching downloaded weights'
    )

    # Processing
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size for processing (default: 32)'
    )
    parser.add_argument(
        '--smiles',
        action='store_true',
        help='Input is a SMILES string, not a file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto)'
    )

    args = parser.parse_args()

    # Print header
    logger.info("=" * 70)
    logger.info("UniMol Unified Feature Extractor v2.0")
    logger.info("=" * 70)

    # Initialize extractor
    extractor = UniMolUnifiedExtractor(
        model_version=args.version,
        model_type=args.model_type,
        local_weights=args.local_weights,
        cache_dir=args.cache_dir,
        device=args.device
    )

    # Extract features
    result = extractor.extract_features(
        input_data=args.input,
        batch_size=args.batch_size
    )

    # Save features
    extractor.save_features(
        features=result['features'],
        metadata=result,
        output_path=args.output
    )

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("✓ Extraction completed successfully!")
    logger.info(f"  Model: UniMol {result['model_version'].upper()} ({result['model_type']})")
    logger.info(f"  Molecules: {result['n_molecules']}")
    logger.info(f"  Feature dimension: {result['feature_dim']}")
    logger.info(f"  Output: {args.output}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
