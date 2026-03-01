# OpenSTL TensorRT 部署指南

本指南介绍如何将 OpenSTL 模型导出为 TensorRT 格式并进行加速推理。

## 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [导出模型](#导出模型)
- [推理](#推理)
- [性能优化](#性能优化)
- [故障排除](#故障排除)

---

## 环境要求

### 硬件要求

| GPU 架构 | computeCapability | 支持的精度 |
|----------|-------------------|------------|
| Pascal   | 6.x               | FP32, FP16 |
| Volta    | 7.x               | FP32, FP16 |
| Turing   | 7.5               | FP32, FP16 |
| Ampere   | 8.x               | FP32, FP16, BF16 |
| Hopper   | 9.x               | FP32, FP16, FP8 |

### 软件要求

```bash
# 安装 TensorRT 和 torch2trt
pip install tensorrt>=8.0
pip install torch2trt>=0.4.0

# 或者使用 NVIDIA 官方容器
docker pull nvcr.io/nvidia/tensorrt:23.10-py3
```

### 验证安装

```bash
python -c "import tensorrt; print(tensorrt.__version__)"
python -c "from torch2trt import torch2trt; print('torch2trt OK')"
```

---

## 快速开始

### 1. 导出 SimVP 模型

```bash
# 导出为 FP16 格式（推荐）
python tools/export_to_trt.py \
    --config configs/mmnist_cifar/simvp/SimVP_gSTA.py \
    --checkpoint work_dirs/simvp_mmnist/checkpoints/best.ckpt \
    --save-dir work_dirs/trt_export \
    --precision fp16 \
    --validate
```

### 2. 运行推理

```bash
python tools/inference_trt.py \
    --trt-model work_dirs/trt_export/model_trt_fp16.pth \
    --config work_dirs/trt_export/export_config.json \
    --input-path data/test_input.npy \
    --output-path results/prediction.npy
```

---

## 导出模型

### 基本导出命令

```bash
python tools/export_to_trt.py \
    --config <配置文件> \
    --checkpoint <检查点文件> \
    --save-dir <导出目录> \
    --precision <精度模式> \
    --batch-size <批大小> \
    --validate
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | 必需 | 模型配置文件路径 |
| `--checkpoint` | 必需 | 训练好的检查点文件 |
| `--save-dir` | `work_dirs/trt_export` | 导出目录 |
| `--precision` | `fp32` | 精度模式：`fp32`, `fp16`, `int8` |
| `--batch-size` | `1` | 导出时的批大小 |
| `--max-batch-size` | `4` | TensorRT 引擎最大批大小 |
| `--validate` | `False` | 是否验证导出模型 |
| `--tolerance` | `1e-4` | 验证容差 |

### 精度模式选择

| 精度 | 速度 | 内存 | 精度损失 | 要求 |
|------|------|------|----------|------|
| FP32 | 1x   | 1x   | 无       | 所有 GPU |
| FP16 | 2-3x | 0.5x | 微小     | compute >= 6.0 |
| INT8 | 4-5x | 0.25x| 小       | compute >= 6.1 + 校准 |

### 导出示例

#### SimVP (Moving MNIST)

```bash
python tools/export_to_trt.py \
    --config configs/mmnist_cifar/simvp/SimVP_gSTA.py \
    --checkpoint work_dirs/simvp_mmnist/checkpoints/best.ckpt \
    --save-dir work_dirs/trt_simvp_mmnist \
    --precision fp16 \
    --batch-size 1 \
    --validate
```

#### SimVP (WeatherBench)

```bash
python tools/export_to_trt.py \
    --config configs/weather/simvp/SimVP_gSTA.py \
    --checkpoint work_dirs/simvp_weather/checkpoints/best.ckpt \
    --save-dir work_dirs/trt_simvp_weather \
    --precision fp16 \
    --batch-size 1 \
    --seq-length 4 \
    --height 64 \
    --width 64 \
    --validate
```

---

## 推理

### Python API

```python
import torch
from torch2trt import TRTModule
import numpy as np

# 加载模型
model = TRTModule()
model.load_state_dict(torch.load('model_trt_fp16.pth'))

# 准备输入
input_data = np.random.randn(1, 10, 1, 64, 64).astype(np.float32)
input_tensor = torch.from_numpy(input_data).cuda()

# 推理
with torch.no_grad():
    output = model(input_tensor)

print(f"Output shape: {output.shape}")
```

### 命令行推理

```bash
python tools/inference_trt.py \
    --trt-model model_trt_fp16.pth \
    --config export_config.json \
    --input-path input.npy \
    --output-path output.npy
```

### 批量推理

```python
import time

# 预热
for _ in range(10):
    _ = model(input_tensor)

# 基准测试
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = model(input_tensor)
torch.cuda.synchronize()

inference_time = (time.time() - start) / 100 * 1000  # ms
print(f"平均推理时间：{inference_time:.2f} ms")
```

---

## 性能优化

### 1. 使用 FP16 精度

```bash
--precision fp16
```

在 Volta 或更新的 GPU 上，FP16 可以提供 2-3 倍加速，精度损失极小。

### 2. 优化工作空间

```bash
--max-workspace-size 2147483648  # 2GB
```

更大的工作空间可以让 TensorRT 选择更优的层实现。

### 3. 动态批处理

对于需要支持动态批大小的场景：

```bash
--max-batch-size 8
```

### 4. 使用 CUDA Graphs

```python
# 启用 CUDA Graphs (PyTorch 1.10+)
model.enable_cuda_graphs = True
```

---

## 故障排除

### 问题 1: torch2trt 导入失败

```bash
# 重新安装
pip uninstall torch2trt
pip install torch2trt --no-cache-dir
```

### 问题 2: FP16 导出失败

```
ERROR: TensorRT could not find an implementation for FP16
```

**解决**: 确认 GPU 支持 FP16（compute capability >= 6.0）

```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
```

### 问题 3: 验证失败（差异过大）

```
Max absolute difference: 0.0015
Validation FAILED (tolerance: 1e-4)
```

**解决**:
1. 增大容差：`--tolerance 1e-3`
2. 使用 FP32 重新导出
3. 检查模型是否有不支持的操作

### 问题 4: 导出后模型过大

```bash
# 使用 INT8 量化（需要校准）
python tools/export_to_trt.py \
    --precision int8 \
    --calib-data calibration_data.npy
```

---

## 支持的模型

| 模型 | FP32 | FP16 | INT8 |
|------|------|------|------|
| SimVP | ✓ | ✓ | ✓ |
| ConvLSTM | ✓ | ✓ | ✓ |
| PredRNN | ✓ | ✓ | ✓ |
| PredRNNv2 | ✓ | ✓ | 待支持 |
| MAU | ✓ | ✓ | 待支持 |
| SwinLSTM | ✓ | ✓ | 待支持 |

---

## 性能对比

以下是在 NVIDIA A100 GPU 上的性能数据（SimVP，batch=1）：

| 精度 | 推理时间 | 加速比 | 内存使用 |
|------|----------|--------|----------|
| PyTorch FP32 | 15.2 ms | 1.0x | 2.1 GB |
| TensorRT FP32 | 10.5 ms | 1.45x | 1.8 GB |
| TensorRT FP16 | 5.8 ms | 2.62x | 1.1 GB |

---

## 参考资源

- [TensorRT 官方文档](https://docs.nvidia.com/deeplearning/tensorrt/)
- [torch2trt GitHub](https://github.com/NVIDIA-AI-IOT/torch2trt)
- [OpenSTL GitHub](https://github.com/chengtan9907/OpenSTL)
