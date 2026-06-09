# 融合时序平滑与风险等级评估的驾驶员分心行为识别及昇腾边缘部署系统

本项目是一个面向 **State Farm Distracted Driver Detection** 数据集的驾驶员分心行为识别工程。基础版使用 **MobileNetV3-Large** 作为主模型，使用 **ResNet18** 作为对比模型，支持按驾驶员 ID 划分训练集、验证集和测试集，并提供训练、评估、混淆矩阵、Grad-CAM、ONNX 导出、昇腾 OM 转换准备以及 PC 摄像头实时识别演示。

实时演示模块不是简单逐帧分类，而是在模型输出后加入了 **EMA 时序平滑、混淆敏感决策过滤、低置信度未知行为拒识、动态风险评分、异常持续时间判断、报警冷却、语音预警和风险等级可视化**，更适合模拟真实车内连续视频流。

本文档只介绍基础版 MobileNetV3-Large 的评估结果，不包含后续自采集摄像头场景微调实验结果。

## 项目框架

```text
E:\ascend
├── configs/
│   └── config.yaml
├── data/
│   ├── statefarm/
│   │   ├── driver_imgs_list.csv
│   │   └── imgs/train/c0 ... c9/
│   └── demo_scene/
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
│   │   ├── decision.py
│   │   ├── risk.py
│   │   └── smoothing.py
│   └── utils/
│       ├── checkpoint.py
│       ├── config.py
│       ├── logger.py
│       └── seed.py
├── scripts/
│   ├── analyze_confusions.py
│   ├── collect_demo_data.py
│   ├── evaluate.py
│   ├── evaluate_manifest.py
│   ├── export_onnx.py
│   ├── finetune_demo_scene.py
│   ├── grad_cam.py
│   ├── prepare_splits.py
│   ├── realtime_demo.py
│   └── train.py
├── outputs/
├── requirements.txt
└── README.md
```

## 数据集

数据集使用 Kaggle 的 **State Farm Distracted Driver Detection**。推荐目录如下：

```text
data/statefarm/
├── driver_imgs_list.csv
└── imgs/
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

类别定义：

| 类别 | 行为 |
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

项目按 `driver_imgs_list.csv` 中的 `subject` 字段进行驾驶员级别划分，避免同一个驾驶员同时出现在 train、val、test 中造成数据泄漏。生成的 manifest 每行格式为：

```text
image_path label subject
```

## 环境

当前项目使用 PyTorch、TorchVision、OpenCV、scikit-learn、matplotlib、seaborn、pyyaml、tqdm、pyttsx3 等依赖。

```powershell
conda activate driver_ai
pip install -r requirements.txt
```

如果在 Windows 上训练时出现 `Couldn't open shared file mapping` 或错误码 `1455`，通常是 DataLoader 多进程共享内存不足。可以降低 `num_workers`，例如使用 `--num-workers 0` 或 `--num-workers 4`。

## 常用命令

生成按驾驶员 ID 划分的数据集：

```powershell
D:\Anaconda_envs\envs\driver_ai\python.exe scripts\prepare_splits.py --config configs\config.yaml
```

训练基础 MobileNetV3-Large：

```powershell
D:\Anaconda_envs\envs\driver_ai\python.exe scripts\train.py --config configs\config.yaml --model mobilenet_v3_large --device cuda
```

训练 ResNet18 对比模型：

```powershell
D:\Anaconda_envs\envs\driver_ai\python.exe scripts\train.py --config configs\config.yaml --model resnet18 --device cuda
```

从 `last.pt` 恢复训练：

```powershell
D:\Anaconda_envs\envs\driver_ai\python.exe scripts\train.py --config configs\config.yaml --model mobilenet_v3_large --device cuda --resume
```

评估基础 MobileNetV3-Large：

```powershell
D:\Anaconda_envs\envs\driver_ai\python.exe scripts\evaluate.py --config configs\config.yaml --model mobilenet_v3_large --checkpoint outputs\checkpoints\mobilenet_v3_large\best.pt --split test --batch-size 64 --num-workers 4 --output-dir outputs\reports\mobilenet_v3_large\test
```

生成 Grad-CAM：

```powershell
D:\Anaconda_envs\envs\driver_ai\python.exe scripts\grad_cam.py --config configs\config.yaml --model mobilenet_v3_large --checkpoint outputs\checkpoints\mobilenet_v3_large\best.pt --image data\statefarm\imgs\train\c0\img_100026.jpg --output outputs\gradcam\mobilenet_overlay.jpg --heatmap-output outputs\gradcam\mobilenet_heatmap.jpg
```

导出 ONNX：

```powershell
D:\Anaconda_envs\envs\driver_ai\python.exe scripts\export_onnx.py --config configs\config.yaml --model mobilenet_v3_large --checkpoint outputs\checkpoints\mobilenet_v3_large\best.pt --output outputs\export\mobilenet_v3_large_driver_distraction.onnx
```

运行 PC 摄像头实时演示：

```powershell
D:\Anaconda_envs\envs\driver_ai\python.exe scripts\realtime_demo.py --config configs\config.yaml --source 0 --checkpoint outputs\checkpoints\mobilenet_v3_large\best.pt --model mobilenet_v3_large --device cuda
```

无窗口快速测试一帧：

```powershell
D:\Anaconda_envs\envs\driver_ai\python.exe scripts\realtime_demo.py --config configs\config.yaml --source data\statefarm\imgs\train\c0\img_100026.jpg --checkpoint outputs\checkpoints\mobilenet_v3_large\best.pt --model mobilenet_v3_large --device cuda --no-window --no-voice --max-frames 1
```

## 基础 MobileNetV3-Large 评估结果

基础版 checkpoint：

```text
outputs/checkpoints/mobilenet_v3_large/best.pt
```

测试集报告路径：

```text
outputs/reports/mobilenet_v3_large/test/
├── classification_report.txt
├── confusion_matrix.csv
├── confusion_matrix.png
└── metrics.json
```

State Farm test 集共有 3866 张图片。基础 MobileNetV3-Large 的整体结果如下：

| 指标 | 数值 |
| --- | ---: |
| Accuracy | 0.8391 |
| Macro Precision | 0.8412 |
| Macro Recall | 0.8390 |
| Macro F1-score | 0.8246 |
| Weighted Precision | 0.8471 |
| Weighted Recall | 0.8391 |
| Weighted F1-score | 0.8281 |
| Loss | 0.5558 |

分类别结果：

| 类别 | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| safe_driving | 0.7887 | 0.2940 | 0.4283 | 381 |
| texting_right | 0.8592 | 0.9734 | 0.9127 | 376 |
| talking_phone_right | 0.9830 | 0.9854 | 0.9842 | 411 |
| texting_left | 0.9400 | 1.0000 | 0.9690 | 407 |
| talking_phone_left | 0.9926 | 1.0000 | 0.9963 | 403 |
| operating_radio | 0.8568 | 0.8111 | 0.8333 | 413 |
| drinking | 0.9943 | 0.8463 | 0.9144 | 410 |
| reaching_behind | 0.7097 | 0.9910 | 0.8271 | 333 |
| hair_and_makeup | 0.7264 | 0.8588 | 0.7871 | 340 |
| talking_to_passenger | 0.5614 | 0.6301 | 0.5938 | 392 |

基础模型已经能较好识别打电话、发短信、喝水等姿态明显的类别；主要问题集中在 `safe_driving` 与 `talking_to_passenger` 等驾驶室内相似姿态之间的混淆。因此实时演示中加入了时序平滑、低置信度拒识和风险持续时间判断，避免单帧误判立即触发报警。

## 摄像头识别推理全流程

实时推理入口是 `scripts/realtime_demo.py`，核心流程位于 `driver_distraction/realtime/camera_demo.py`。

1. 读取配置

   程序加载 `configs/config.yaml` 中的 `realtime` 配置，包括摄像头编号、checkpoint、模型名、输入尺寸、置信度阈值、EMA 参数、风险阈值、报警冷却时间和语音设置。

2. 加载模型

   `load_realtime_model` 根据配置构建 MobileNetV3-Large，并从 `outputs/checkpoints/mobilenet_v3_large/best.pt` 加载权重。模型切换到 `eval` 模式，推理时使用 `torch.inference_mode()`。

3. 打开视频源

   OpenCV 使用 `cv2.VideoCapture` 打开摄像头、视频文件或图片源。配置中可以指定摄像头宽度、高度和 FPS。

4. 图像预处理

   每帧 BGR 图像先转为 RGB，再转为 PIL 图像，使用实时推理 transform：

   ```text
   Resize(256) -> CenterCrop(224) -> ToTensor -> ImageNet Normalize
   ```

5. 模型前向推理

   MobileNetV3-Large 输出 10 类 logits，经过 softmax 得到每个驾驶行为类别的概率分布。

6. EMA 时序平滑

   `EMASmoother` 对连续帧概率做指数滑动平均：

   ```text
   p_smooth[t] = alpha * p_raw[t] + (1 - alpha) * p_smooth[t - 1]
   ```

   这样可以降低单帧抖动。`alpha` 越大响应越快，越小越稳定。当前配置默认使用 `alpha=0.35`，并支持长时间断帧后重置平滑状态。

7. 混淆敏感决策过滤

   `TemporalDecisionFilter` 会检查 top1 与 top2 的概率间隔。如果预测落在已知易混淆类别对中，并且 margin 较小，则保持已有稳定类别，要求连续若干帧确认后才切换标签。当前配置特别关注：

   ```text
   safe_driving <-> talking_to_passenger
   talking_to_passenger <-> reaching_behind
   texting_right <-> reaching_behind
   operating_radio <-> reaching_behind
   operating_radio <-> hair_and_makeup
   drinking <-> texting_right
   hair_and_makeup <-> reaching_behind
   ```

8. 低置信度拒识

   如果最终置信度低于 `confidence_threshold`，系统不会强行输出某个已知类别，而是输出 `unknown`。这可以减少现实摄像头角度、背景、光照与训练集差异较大时的误报。

9. 动态风险评分

   `RiskAssessor` 根据类别风险权重和置信度计算瞬时风险，再用衰减因子更新整体风险分数。安全驾驶风险权重为 0，发短信、伸手到后座等高危行为权重较高。

10. 异常持续时间判断

    系统不会因为一帧异常就报警。只有当非安全行为持续超过 `abnormal_hold_seconds`，且风险等级达到 medium 或 high 时，才认为满足报警条件。

11. 报警冷却与语音预警

    `AlarmManager` 使用 `alarm_cooldown_seconds` 控制报警冷却，避免连续重复播报。语音预警使用 `pyttsx3`，可异步播放，也可以通过 `--no-voice` 关闭。

12. 可视化输出

    `draw_dashboard` 在画面顶部显示稳定后的行为标签、置信度、原始预测、margin、风险等级、风险分数、异常持续时间、冷却剩余时间和 FPS；底部绘制风险进度条。可选参数 `--save-video` 可以保存演示视频。

## 文件夹说明

| 路径 | 作用 |
| --- | --- |
| `configs/` | 全局配置目录，集中管理数据路径、训练参数、评估参数、Grad-CAM、ONNX、昇腾转换和实时演示参数。 |
| `data/statefarm/` | State Farm 数据集存放目录，包含元数据 CSV 和原始训练图片。 |
| `data/demo_scene/` | 自采集演示场景数据目录，用于后续摄像头角度适配或小样本微调。基础版评估不使用该目录。 |
| `deploy/` | 昇腾部署说明和 ATC 转 OM 模板脚本。 |
| `driver_distraction/` | 项目核心 Python 包，包含数据、模型、训练评估、可解释性、实时推理、部署和工具代码。 |
| `outputs/` | 训练 checkpoint、数据划分、评估报告、Grad-CAM 图像、ONNX 导出文件等输出目录。 |
| `scripts/` | 命令行入口脚本目录，负责训练、评估、导出、演示和辅助分析。 |

## 代码文件说明

### 配置与常量

| 文件 | 功能 |
| --- | --- |
| `configs/config.yaml` | 项目主配置文件，包含数据集路径、输入尺寸、类别名、训练超参数、评估 checkpoint、Grad-CAM、ONNX、昇腾 ATC 参数和实时演示参数。 |
| `driver_distraction/constants.py` | State Farm 10 类行为名称、类别到索引、索引到类别的常量映射。 |

### 数据模块

| 文件 | 功能 |
| --- | --- |
| `driver_distraction/data/splits.py` | 读取 `driver_imgs_list.csv`，按驾驶员 ID 生成 train/val/test 划分，保存 split JSON 和 manifest。 |
| `driver_distraction/data/statefarm.py` | `StateFarmDataset` 和 DataLoader 构建逻辑。支持 manifest 格式 `image_path label subject`，返回 `image_tensor, label, image_path, subject`。 |
| `driver_distraction/data/transforms.py` | 图像预处理。训练阶段使用 Resize、RandomResizedCrop、ColorJitter、RandomRotation、Normalize；验证、测试和实时推理使用 Resize、CenterCrop、Normalize。没有使用 RandomHorizontalFlip，避免左右手行为类别被翻转后标签错误。 |

### 模型模块

| 文件 | 功能 |
| --- | --- |
| `driver_distraction/models/factory.py` | 模型工厂，支持构建 `mobilenet_v3_large` 和 `resnet18`，替换分类头，支持预训练权重、dropout 和冻结 backbone。 |

### 训练与评估模块

| 文件 | 功能 |
| --- | --- |
| `driver_distraction/engine/trainer.py` | 训练主循环，包含 optimizer、scheduler、loss、AMP、checkpoint 保存、断点恢复、history.csv 写入和验证集 best 模型选择。 |
| `driver_distraction/engine/evaluator.py` | 模型评估函数，输出 loss、accuracy、真实标签、预测标签和 softmax 概率。 |
| `driver_distraction/engine/metrics.py` | 分类报告、Precision、Recall、F1-score、混淆矩阵图片和 CSV 保存工具。 |

### Grad-CAM 模块

| 文件 | 功能 |
| --- | --- |
| `driver_distraction/explain/grad_cam.py` | Grad-CAM 实现，自动选择 CNN 最后一层特征层，生成热力图和叠加图，用于观察模型关注区域。 |

### 实时推理模块

| 文件 | 功能 |
| --- | --- |
| `driver_distraction/realtime/smoothing.py` | EMA 时序平滑，对连续帧概率进行指数滑动平均，减少识别抖动。 |
| `driver_distraction/realtime/decision.py` | 混淆敏感的时序决策过滤，对易混淆类别要求连续帧确认后再切换稳定标签。 |
| `driver_distraction/realtime/risk.py` | 动态风险评分、风险等级划分、异常行为持续时间统计和报警触发条件判断。 |
| `driver_distraction/realtime/alarm.py` | 语音预警和报警冷却管理，支持 `pyttsx3` 异步播报。 |
| `driver_distraction/realtime/camera_demo.py` | PC 摄像头实时演示核心流程：取帧、预处理、模型推理、时序平滑、拒识、风险评估、报警、可视化和视频保存。 |

### 昇腾部署模块

| 文件 | 功能 |
| --- | --- |
| `driver_distraction/deploy/ascend.py` | 根据配置生成 Ascend ATC 转换命令，便于将 ONNX 转为 OM。 |
| `deploy/README_ascend.md` | 昇腾部署说明，描述 ONNX 到 OM 的转换准备。 |
| `deploy/ascend_atc_template.sh` | ATC 命令模板脚本，用于在 Ascend Toolkit 环境中转换 OM 模型。 |

### 工具模块

| 文件 | 功能 |
| --- | --- |
| `driver_distraction/utils/checkpoint.py` | checkpoint 保存与加载。 |
| `driver_distraction/utils/config.py` | YAML 配置读取和保存。 |
| `driver_distraction/utils/logger.py` | 日志工具预留。 |
| `driver_distraction/utils/seed.py` | 随机种子设置，提升训练复现性。 |

### 脚本入口

| 文件 | 功能 |
| --- | --- |
| `scripts/prepare_splits.py` | 生成驾驶员级别 train/val/test 划分和 manifest 文件。 |
| `scripts/train.py` | 训练入口，支持 MobileNetV3-Large、ResNet18、GPU、AMP、断点恢复和命令行覆盖超参数。 |
| `scripts/evaluate.py` | 标准 train/val/test 评估入口，输出分类报告、混淆矩阵和 metrics.json。 |
| `scripts/evaluate_manifest.py` | 任意 manifest 评估入口，适合评估自采集小数据或局部类别数据。 |
| `scripts/analyze_confusions.py` | 从混淆矩阵 CSV 中提取主要误判类别对。 |
| `scripts/grad_cam.py` | Grad-CAM 命令行入口，生成热力图和叠加图。 |
| `scripts/export_onnx.py` | ONNX 导出入口，为后续昇腾 ATC 转 OM 做准备。 |
| `scripts/realtime_demo.py` | PC 摄像头或视频实时演示入口。 |
| `scripts/collect_demo_data.py` | 自采集演示场景脚本，可选择 C0-C9 类别，从摄像头按固定间隔采集图片。 |
| `scripts/finetune_demo_scene.py` | 自采集场景小样本微调脚本，可混合原始训练集 replay 降低遗忘风险。基础版评估不使用该脚本。 |

## ONNX 与昇腾 OM 部署流程

基础流程如下：

1. 使用 `scripts/export_onnx.py` 将 PyTorch checkpoint 导出为 ONNX。
2. 在安装 Ascend Toolkit 的环境中运行 ATC，将 ONNX 转为 OM。
3. 后续可接入 ACL、MindX SDK 或 StreamManager 推理流程。

导出 ONNX：

```powershell
D:\Anaconda_envs\envs\driver_ai\python.exe scripts\export_onnx.py --config configs\config.yaml --model mobilenet_v3_large --checkpoint outputs\checkpoints\mobilenet_v3_large\best.pt --output outputs\export\mobilenet_v3_large_driver_distraction.onnx
```

ATC 转换命令模板见：

```text
deploy/ascend_atc_template.sh
```

核心参数来自 `configs/config.yaml` 的 `ascend` 配置：

```yaml
ascend:
  om_path: outputs/export/driver_distraction.om
  soc_version: Ascend310P3
  input_shape: "input:1,3,224,224"
  precision_mode: allow_fp32_to_fp16
```

## 输出文件

常见输出目录如下：

| 路径 | 内容 |
| --- | --- |
| `outputs/splits/` | 驾驶员级别 split JSON 和 train/val/test manifest。 |
| `outputs/checkpoints/mobilenet_v3_large/` | 基础 MobileNetV3-Large 的 `best.pt`、`last.pt`、`history.csv` 和训练配置副本。 |
| `outputs/checkpoints/resnet18/` | ResNet18 对比模型 checkpoint。 |
| `outputs/reports/mobilenet_v3_large/test/` | 基础 MobileNetV3-Large 测试集分类报告、混淆矩阵和指标 JSON。 |
| `outputs/gradcam/` | Grad-CAM 热力图和叠加图。 |
| `outputs/export/` | ONNX 和后续 OM 模型导出目录。 |

## 现阶段基础版结论

基础 MobileNetV3-Large 具有模型轻量、推理速度友好、便于 ONNX 和边缘部署的优点，在 State Farm 测试集上达到 `0.8391` 的 Accuracy 和 `0.8246` 的 Macro F1-score。它适合作为后续昇腾边缘部署和实时风险预警系统的主干模型。

当前主要不足是现实摄像头角度、驾驶室背景和 State Farm 数据集分布可能存在差异，且 `safe_driving` 与 `talking_to_passenger` 等相似姿态容易混淆。因此项目在基础分类模型之外加入了时序平滑、决策过滤、拒识和风险持续时间判断，用于提升实时演示阶段的稳定性和可解释性。
