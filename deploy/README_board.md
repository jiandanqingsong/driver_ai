# 昇腾开发板部署流程

本文档说明如何把本项目通过 GitHub 下载到昇腾开发板，并在开发板上使用 Ascend Toolkit 将 ONNX 模型转换为 OM 模型。

## 1. GitHub 仓库建议保留的内容

仓库中建议保留：

```text
configs/
deploy/
driver_distraction/
scripts/
README.md
requirements.txt
```

用于板端转换的 ONNX 模型放在：

```text
deploy/models/mobilenet_v3_large_demo_finetune_driver_distraction.onnx
```

仓库中不建议提交：

```text
data/statefarm/
data/demo_scene/
outputs/
runs/
wandb/
*.pt
*.pth
*.om
```

原因：

- `data/statefarm/` 是训练数据集，体积大，开发板推理不需要。
- `data/demo_scene/` 是本地采集数据，属于实验数据，部署不需要。
- `outputs/` 包含训练日志、报告、checkpoint 和临时导出文件，容易污染仓库。
- `.pt/.pth` 是 PyTorch 训练权重，板端 OM 推理不需要。
- `.om` 建议在目标开发板或同型号环境中用 ATC 生成，不建议提交。

如果不想把 ONNX 直接提交到 GitHub，可以把它放到 GitHub Release、对象存储或 U 盘中，然后在开发板 clone 项目后手动复制到 `deploy/models/`。

## 2. 在 PC 端准备 ONNX

如果还没有 ONNX，先在训练机器上导出微调后的 MobileNetV3-Large：

```powershell
D:\Anaconda_envs\envs\driver_ai\python.exe scripts\export_onnx.py --config configs\config.yaml --model mobilenet_v3_large --checkpoint outputs\checkpoints\mobilenet_v3_large_demo_finetune\best.pt --output deploy\models\mobilenet_v3_large_demo_finetune_driver_distraction.onnx
```

确认文件存在：

```powershell
dir deploy\models\mobilenet_v3_large_demo_finetune_driver_distraction.onnx
```

然后提交代码和部署模型：

```bash
git add .gitignore README.md configs deploy driver_distraction scripts requirements.txt
git commit -m "Add Ascend board deployment files"
git push
```

如果 ONNX 使用 Git LFS 管理，先执行：

```bash
git lfs track "deploy/models/*.onnx"
git add .gitattributes deploy/models/*.onnx
git commit -m "Track deployment ONNX with Git LFS"
```

## 3. 在开发板上下载项目

登录开发板后执行：

```bash
git clone https://github.com/<your-name>/<your-repo>.git
cd <your-repo>
```

如果模型用 Git LFS：

```bash
git lfs install
git lfs pull
```

确认 ONNX 已经在开发板上：

```bash
ls -lh deploy/models/mobilenet_v3_large_demo_finetune_driver_distraction.onnx
```

## 4. 激活 Ascend Toolkit 环境

常见路径如下，按开发板实际安装位置选择一个：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

或：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

检查 ATC：

```bash
which atc
atc --version
```

如果 `atc` 找不到，说明 Toolkit 环境变量没有生效，需要检查 `set_env.sh` 路径。

## 5. ONNX 转 OM

默认转换命令：

```bash
bash deploy/ascend_atc_template.sh
```

默认输入：

```text
deploy/models/mobilenet_v3_large_demo_finetune_driver_distraction.onnx
```

默认输出：

```text
deploy/models/mobilenet_v3_large_demo_finetune_driver_distraction.om
```

脚本默认参数：

```bash
SOC_VERSION=Ascend310P3
INPUT_SHAPE=input:1,3,224,224
PRECISION_MODE=allow_fp32_to_fp16
```

如果开发板芯片型号不是 `Ascend310P3`，转换时覆盖 `SOC_VERSION`：

```bash
SOC_VERSION=Ascend310B4 bash deploy/ascend_atc_template.sh
```

也可以手动指定输入 ONNX 和输出 OM：

```bash
bash deploy/ascend_atc_template.sh \
  deploy/models/mobilenet_v3_large_demo_finetune_driver_distraction.onnx \
  deploy/models/mobilenet_v3_large_demo_finetune_driver_distraction.om
```

转换成功后检查：

```bash
ls -lh deploy/models/*.om
```

## 6. 板端推理输入输出约定

模型输入：

```text
name: input
shape: 1,3,224,224
format: NCHW
dtype: FP32
```

预处理必须和训练/导出保持一致：

```text
BGR/RGB 图像 -> RGB
Resize(256)
CenterCrop(224)
ToTensor
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

模型输出：

```text
name: logits
shape: 1,10
```

后处理：

```text
softmax(logits) -> 10 类概率 -> argmax 得到类别
```

类别顺序：

```text
0 safe_driving
1 texting_right
2 talking_phone_right
3 texting_left
4 talking_phone_left
5 operating_radio
6 drinking
7 reaching_behind
8 hair_and_makeup
9 talking_to_passenger
```

## 7. 验证 OM 模型

如果开发板安装了 `ais_bench`，可先做静态推理验证：

```bash
python3 -m ais_bench --model deploy/models/mobilenet_v3_large_demo_finetune_driver_distraction.om --output outputs/ais_bench
```

不同开发板镜像中 `ais_bench` 命令可能是 `ais_bench`、`benchmark` 或未预装。若未安装，可以先只完成 ATC 转换，后续使用 ACL Python、ACL C++ 或 MindX SDK 接入 `.om`。

## 8. 关于 Web 实时监控

当前 `scripts/web_demo.py` 是 PC/PyTorch 版 Web 演示入口，加载的是 `.pt` checkpoint，并使用 PyTorch 做推理。它适合在训练机或普通 PC 上展示完整的 EMA、风险评分、报警冷却和网页统计面板。

在昇腾开发板上使用 `.om` 模型实时推理时，推荐路线是：

1. 使用本文档完成 ONNX 到 OM 转换。
2. 用 ACL Python、ACL C++ 或 MindX SDK 编写 OM 推理封装。
3. 复用项目中的后处理逻辑：
   - `driver_distraction/realtime/smoothing.py`
   - `driver_distraction/realtime/decision.py`
   - `driver_distraction/realtime/risk.py`
   - `driver_distraction/realtime/alarm.py`
4. 将 `web_demo.py` 中的 PyTorch `predict_frame` 替换为 OM 推理结果。

也就是说，板端最终运行时不需要训练代码、State Farm 数据集、PyTorch checkpoint 或训练日志，只需要源码、配置、ONNX/OM 模型和 OM 推理入口。

## 9. 常见问题

### atc 找不到

先执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
which atc
```

### SOC_VERSION 不匹配

查看开发板芯片型号或 Toolkit 文档，然后覆盖环境变量：

```bash
SOC_VERSION=Ascend310B4 bash deploy/ascend_atc_template.sh
```

### 输入 shape 报错

本项目导出的 ONNX 使用动态 batch，但板端转换推荐固定 batch：

```bash
INPUT_SHAPE=input:1,3,224,224 bash deploy/ascend_atc_template.sh
```

### 推理结果类别错位

检查后处理类别顺序是否和 `configs/config.yaml` 中 `data.class_names` 完全一致。
