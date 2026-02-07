# UniMol Feature Extractor - Unified Framework (v1 + v2)

## 🎯 Project Overview

A **unified framework** for extracting molecular representations using **both UniMol v1 and v2** models with automatic weight management.

## 📦 What's New

### ✅ Completed Components

1. **Smart Weight Manager** (`weight_manager.py`)
   - Automatic local weight detection
   - Download from HuggingFace if not found locally
   - Caching for offline use
   - Support for both v1 and v2 models

2. **Unified Extractor** (`unified_extractor.py`)
   - Single interface for UniMol v1 and v2
   - Automatic version selection
   - Batch processing
   - GPU acceleration

3. **Weight Management Strategy**
   ```
   Priority:
   1. Local weights (if specified)
   2. Cached weights
   3. Auto-download from HuggingFace
   ```

## 🚀 Usage

### UniMol v1

```bash
# Auto-download weights
python unified_extractor.py \
    --version v1 \
    --model-type molecule_all_h \
    --input molecules.sdf \
    --output features.npz

# Use local weights
python unified_extractor.py \
    --version v1 \
    --model-type molecule_all_h \
    --input molecules.sdf \
    --output features.npz \
    --local-weights /path/to/mol_pre_all_h_220816.pt
```

### UniMol v2

```bash
# Use 1.1B model (auto-download)
python unified_extractor.py \
    --version v2 \
    --model-type 1.1B \
    --input molecules.sdf \
    --output features.npz

# Use 310M model
python unified_extractor.py \
    --version v2 \
    --model-type 310m \
    --input molecules.sdf \
    --output features.npz
```

## 📋 Supported Models

### UniMol v1
- `molecule_all_h` - General molecules with all hydrogens
- `oled_no_h` - OLED compounds without hydrogens

### UniMol v2
- `84m` - 84M parameter model
- `164m` - 164M parameter model
- `310m` - 310M parameter model
- `570m` - 570M parameter model
- `1.1B` - 1.1B parameter model

## 🔧 Weight Management

### Automatic Behavior

The extractor automatically:
1. Checks if local weights are provided
2. Checks cache directory (`~/.cache/unimol_weights/`)
3. Downloads from HuggingFace if needed

### Cache Location

```bash
# Default cache
~/.cache/unimol_weights/

# Custom cache
python unified_extractor.py --cache-dir /path/to/cache ...
```

### Clear Cache

```python
from weight_manager import UniMolWeightManager

manager = UniMolWeightManager()
manager.clear_cache()  # Clear all
manager.clear_cache('unimolv1')  # Clear v1 only
```

## 📊 Output Format

```python
data = np.load('features.npz')
features = data['features']  # (n_molecules, feature_dim)
metadata = {
    'model_version': 'v1' or 'v2',
    'model_type': 'molecule_all_h' or '84m', etc.
    'feature_dim': 512 (v1) or 1536 (v2 1.1B),
    'n_molecules': int,
    'extraction_date': str
}
```

## 🔗 Dependencies

### Required

```bash
# Core
pip install torch numpy rdkit scikit-learn

# UniMol v1
pip install unimol_tools

# UniMol v2 (from source)
git clone https://github.com/dptech/unimol.git
cd unimol
pip install -e .

# Weight download
pip install huggingface-hub
```

### Environment Variables

```bash
# UniMol v1 path (optional)
export UNIMOL_PATH="/path/to/unimol"

# UniMol v2 path (optional)
export UNIMOL2_PATH="/path/to/unimol2"
```

## 📁 Project Structure

```
UniMol Feature Extractor/
├── weight_manager.py          # Smart weight management
├── unified_extractor.py       # Unified v1+v2 extractor
├── unimol_extractor.py        # Original v1 extractor (kept)
├── examples.py                # Usage examples
├── requirements.txt           # Dependencies
├── DEPENDENCIES.md            # Detailed dependency guide
├── README.md                  # Full documentation
└── README_V2.md              # This file
```

## 🎓 Comparison: v1 vs v2

| Feature | UniMol v1 | UniMol v2 |
|---------|-----------|-----------|
| Feature Dim | 512 | 768-1536 |
| Model Sizes | 1 (~181M) | 5 (84m-1.1B) |
| Training Data | Molecules, OLED | Larger, more diverse |
| Speed | Faster | Slower (larger models) |
| Performance | Good | Better (state-of-art) |

## 💡 Recommendations

### When to Use v1
- Limited GPU memory
- Need faster processing
- General molecular representation

### When to Use v2
- Need best performance
- Have sufficient GPU memory
- State-of-the-art results required

## 🔄 Migration from v1

```bash
# Old command
python unimol_extractor.py --input molecules.sdf --output features.npz

# New command (same functionality)
python unified_extractor.py --version v1 --input molecules.sdf --output features.npz
```

## 🚧 Status

- [x] Weight manager with auto-download
- [x] Unified v1+v2 interface
- [x] CLI for both versions
- [ ] Complete v2 batch processing
- [ ] Full documentation update
- [ ] Test suite
- [ ] Examples for v2

## 📝 Next Steps

1. Complete UniMol v2 implementation
2. Add comprehensive tests
3. Update all documentation
4. Create migration guide
5. Add performance benchmarks

---

**Version**: 2.0.0 (Beta)
**Status**: Active Development
