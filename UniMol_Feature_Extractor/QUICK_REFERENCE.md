# 快速参考指南 - UniMol 特征提取器

## 🎯 快速选择

### 只需要 UniMol v1？
```bash
cd "UniMol1 Feature Extractor_backup/"
python unimol_extractor.py --input molecules.sdf --output features.npz
```

### 需要在 v1 和 v2 之间选择？
```bash
cd "UniMol Feature Extractor/"

# v1 - 快速，512维
python unified_extractor.py --version v1 --input molecules.sdf --output v1.npz

# v2 - 强大，1536维 (1.1B模型)
python unified_extractor.py --version v2 --model-type 1.1B --input molecules.sdf --output v2.npz
```

## 📋 常用命令

### UniMol v1

```bash
# 基本
python unimol_extractor.py -i molecules.sdf -o features.npz

# 自定义权重
python unimol_extractor.py -i molecules.sdf -o features.npz \
    -w /path/to/weights.pt -d /path/to/dict.txt

# SMILES 输入
python unimol_extractor.py -i "CCO" -o ethanol.npz --smiles

# GPU 加速
python unimol_extractor.py -i molecules.sdf -o features.npz --device cuda
```

### 统一提取器 (v1+v2)

```bash
# v1 基本用法
python unified_extractor.py --version v1 --input molecules.sdf --output v1.npz

# v2 不同模型
python unified_extractor.py --version v2 --model-type 84m --input molecules.sdf --output 84m.npz
python unified_extractor.py --version v2 --model-type 310m --input molecules.sdf --output 310m.npz
python unified_extractor.py --version v2 --model-type 1.1B --input molecules.sdf --output 1.1B.npz

# 本地权重
python unified_extractor.py --version v1 --input molecules.sdf --output features.npz \
    --local-weights /path/to/weights.pt

# 自定义缓存目录
python unified_extractor.py --version v1 --input molecules.sdf --output features.npz \
    --cache-dir /custom/cache/path
```

## 🔍 模型选择

### UniMol v1
- `molecule_all_h` - 通用分子（推荐）
- `oled_no_h` - OLED 分子

### UniMol v2
- `84m` - 快速，768维
- `164m` - 平衡
- `310m` - 推荐，1024维
- `570m` - 高性能
- `1.1B` - 最佳，1536维

## 💾 权重管理

### 自动行为
1. 检查本地权重（如果指定）
2. 检查缓存 `~/.cache/unimol_weights/`
3. 从 HuggingFace 下载

### 缓存管理
```python
from weight_manager import UniMolWeightManager

# 清除所有缓存
manager = UniMolWeightManager()
manager.clear_cache()

# 清除特定版本
manager.clear_cache('unimolv1')
manager.clear_cache('unimolv2')
```

## 📊 特征维度

| 版本 | 模型 | 维度 |
|------|------|------|
| v1 | 所有 | 512 |
| v2 | 84m | 768 |
| v2 | 164m | 896 |
| v2 | 310m | 1024 |
| v2 | 570m | 1280 |
| v2 | 1.1B | 1536 |

## 🚨 故障排除

### "Cannot find UniMol"
```bash
# 安装 UniMol v1
pip install unimol_tools

# 或设置路径
export UNIMOL_PATH="/path/to/unimol"
```

### "Weights not found"
```bash
# 脚本会自动下载
# 或手动指定
--local-weights /path/to/weights.pt
```

### "CUDA out of memory"
```bash
# 减小批大小
--batch-size 16

# 或使用 CPU
--device cpu

# 或使用更小的模型
--model-type 84m  # v2
```

## 📖 完整文档

- `README.md` - UniMol v1 完整文档
- `README_V2.md` - 统一框架说明
- `DEPENDENCIES.md` - 依赖详解
- `PROJECT_SUMMARY.md` - 项目总结

## 🎯 推荐使用

**新手**: 使用 v1 提取器（简单）
**灵活性**: 使用统一提取器（v1+v2）
**性能**: 使用 v2 1.1B 模型
**速度**: 使用 v2 84m 模型

---

**快速参考** - 保存此文件以备查阅！
