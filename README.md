# 融合时序平滑与风险等级评估的驾驶员分心行为识别及昇腾边缘部署系统

这是一个面向 **State Farm Distracted Driver Detection** 数据集的 Python/PyTorch 工程骨架。项目主模型采用 **MobileNetV3-Large**，对比模型采用 **ResNet18**，支持按驾驶员 ID 划分 train/val/test、训练、评估、混淆矩阵、Grad-CAM、ONNX 导出，以及后续 Ascend ATC 转 OM 模型部署。实时摄像头演示模块预留了 EMA 时序平滑、动态风险评分、异常持续时间判断、低置信度未知行为拒识、语音预警、报警冷却和风险等级可视化。

## 推荐目录结构

```text
.
├── configs/
│   └── config.yaml
├── data/
│   └── README.md
├── deploy/
│   ├── README_ascend.md
│   └── ascend_atc_template.sh
├── driver_distraction/
│   ├── constants.py
│   ├── data/
│   │   ├── splits.py
│   │   ├── statefarm.py
│   │   └── transforms.py
│   ├── deploy/
│   │   └── ascend.py
│   ├── engine/
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   └── trainer.py
│   ├── explain/
│   │   └── grad_cam.py
│   ├── models/
│   │   └── factory.py
│   ├── realtime/
│   │   ├── alarm.py
│   │   ├── camera_demo.py
│   │   ├── risk.py
│   │   └── smoothing.py
│   └── utils/
│       ├── checkpoint.py
│       ├── config.py
│       ├── logger.py
│       └── seed.py
├── scripts/
│   ├── evaluate.py
│   ├── export_onnx.py
│   ├── grad_cam.py
│   ├── prepare_splits.py
│   ├── realtime_demo.py
│   └── train.py
└── requirements.txt
```

## 环境安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell 可使用：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 数据集放置

从 Kaggle 下载 State Farm Distracted Driver Detection 后，建议整理为：

```text
data/statefarm/
├── driver_imgs_list.csv
└── train/
    ├── c0/
    ├── c1/
    ├── c2/
    ├── c3/
    ├── c4/
    ├── c5/
    ├── c6/
    ├── c7/
    ├── c8/
    └── c9/
```

类别含义：

| 原始类别 | 行为 |
| --- | --- |
| c0 | safe_driving |
| c1 | texting_right |
| c2 | talking_phone_right |
| c3 | texting_left |
| c4 | talking_phone_left |
| c5 | operating_radio |
| c6 | drinking |
| c7 | reaching_behind |
| c8 | hair_and_makeup |
| c9 | talking_to_passenger |

## 常用命令

生成按驾驶员 ID 划分的数据 split：

```bash
python scripts/prepare_splits.py --config configs/config.yaml
```

训练主模型 MobileNetV3-Large：

```bash
python scripts/train.py --config configs/config.yaml --model mobilenet_v3_large
```

训练对比模型 ResNet18：

```bash
python scripts/train.py --config configs/config.yaml --model resnet18
```

评估并输出分类报告和混淆矩阵：

```bash
python scripts/evaluate.py --config configs/config.yaml --split test
```

生成 Grad-CAM 可视化：

```bash
python scripts/grad_cam.py --config configs/config.yaml --image data/statefarm/train/c1/img_1.jpg
```

导出 ONNX：

```bash
python scripts/export_onnx.py --config configs/config.yaml
```

实时摄像头演示：

```bash
python scripts/realtime_demo.py --config configs/config.yaml --source 0
```

## 昇腾 OM 部署流程

1. 先用 `scripts/export_onnx.py` 导出 ONNX。
2. 在 Ascend Toolkit 环境中执行 `deploy/ascend_atc_template.sh` 或使用 `driver_distraction.deploy.ascend.build_atc_command` 生成 ATC 命令。
3. 生成 `.om` 后，后续可接入 ACL / MindX SDK / StreamManager 推理流程。

示例：

```bash
bash deploy/ascend_atc_template.sh
```

## 主要文件功能说明

| 文件 | 功能 |
| --- | --- |
| `configs/config.yaml` | 全局配置，包含数据路径、训练参数、评估输出、Grad-CAM、ONNX、Ascend 和实时演示参数。 |
| `requirements.txt` | Python 依赖列表。 |
| `driver_distraction/constants.py` | State Farm 类别映射与类别名称常量。 |
| `driver_distraction/data/splits.py` | 基于驾驶员 ID 的 train/val/test 划分逻辑，可保存和读取 split JSON。 |
| `driver_distraction/data/statefarm.py` | State Farm 数据集类与 DataLoader 构建入口。 |
| `driver_distraction/data/transforms.py` | 训练、验证、测试和实时推理图像预处理。 |
| `driver_distraction/models/factory.py` | MobileNetV3-Large 与 ResNet18 模型工厂。 |
| `driver_distraction/engine/trainer.py` | 训练循环、验证循环、优化器和调度器构建。 |
| `driver_distraction/engine/evaluator.py` | 模型评估，输出 loss、accuracy、预测标签和概率。 |
| `driver_distraction/engine/metrics.py` | 分类报告与混淆矩阵绘制。 |
| `driver_distraction/explain/grad_cam.py` | Grad-CAM 特征图提取、热力图生成和叠加可视化。 |
| `driver_distraction/realtime/smoothing.py` | EMA 时序平滑，稳定连续帧预测结果。 |
| `driver_distraction/realtime/risk.py` | 动态风险评分、风险等级、异常持续时间判断。 |
| `driver_distraction/realtime/alarm.py` | 语音预警和报警冷却管理。 |
| `driver_distraction/realtime/camera_demo.py` | 实时摄像头推理、拒识、风险可视化和报警主流程。 |
| `driver_distraction/deploy/ascend.py` | Ascend ATC 命令生成辅助函数。 |
| `driver_distraction/utils/*.py` | 配置加载、随机种子、日志和 checkpoint 工具。 |
| `scripts/prepare_splits.py` | 生成驾驶员级别数据划分文件。 |
| `scripts/train.py` | 训练入口，支持选择 MobileNetV3-Large 或 ResNet18。 |
| `scripts/evaluate.py` | 评估入口，生成分类报告和混淆矩阵。 |
| `scripts/grad_cam.py` | 单张图像 Grad-CAM 可视化入口。 |
| `scripts/export_onnx.py` | ONNX 导出入口。 |
| `scripts/realtime_demo.py` | 实时摄像头演示入口。 |
| `deploy/README_ascend.md` | Ascend OM 转换说明。 |
| `deploy/ascend_atc_template.sh` | ATC 转 OM 命令模板。 |

## 后续建议

- 补充实验记录表：比较 MobileNetV3-Large 与 ResNet18 在 driver split 下的 accuracy、macro-F1、推理延迟和模型大小。
- 在 Ascend 环境中增加 OM 推理 wrapper，并记录 ONNX 与 OM 输出一致性误差。
- 将实时演示中的风险策略参数固定为论文/报告中的可解释表格。
