# MobileNetV4 训练改进策略记录

> 记录时间：2026-06-10
> 目标：解决 MobileNetV4（timm `mobilenetv4_conv_medium`）在 State Farm 驾驶员划分数据集上训练效果不佳的问题。

本文档独立于主 `README.md`，专门记录本次针对 MobileNetV4 的问题诊断、改进策略、代码改动和验证方式。基础 MobileNetV3-Large 的已有 checkpoint 和结论不受影响。

---

## 1. 问题诊断

MobileNetV4 首次训练（12 epoch，沿用 V3 的超参）的 `history.csv` 关键数据：

| 现象 | 数据 | 含义 |
| --- | --- | --- |
| 训练集飙升 | `train_acc` 0.73 → **0.98**，`train_loss` → 0.63 | 训练集几乎被背下来 |
| 验证集卡住 | best `val_macro_f1=0.866` 出现在 **第 2 个 epoch**，后 10 轮再未超越 | 有效学习只发生在前 2 轮 |
| 验证集剧烈震荡 | `val_loss`：ep3=**2.71**、ep6=1.62、ep8=**7.22**、ep12=1.95 | 学习率过高，AdamW 在部分 batch 上发散 |
| 泛化差距大 | train 0.98 vs val ~0.80（约 18 个点） | 典型过拟合 |

**关键判断**：State Farm 按司机（subject）划分，val/test 是训练中从未出现的人。`train_acc≈1.0` 说明模型记住了训练司机的长相，而不是学到通用的分心行为特征。

> 结论：问题不是"学得不够"，而是 **过拟合 + 训练不稳定**。对策是 **正则化 + 稳定化 + 提升跨司机泛化**，而非延长训练。

与 MobileNetV3-Large 对比：V3 曲线平滑，best 在 ep8=0.874；V4 容量更大、收敛更快，第 2 轮即到顶后一路震荡。

---

## 2. 改进策略（按性价比排序）

本次采用 **稳健组合**（不引入权重 EMA）：

| # | 策略 | 作用 | 落地方式 |
| --- | --- | --- | --- |
| ① | **随机深度 drop_path** | timm 模型核心正则项，直接收窄 train/val 差距 | `build_mobilenet_v4` 传入 `drop_path_rate=0.1` |
| ② | **降学习率 + warmup** | 消除 `val_loss` 爆炸 | 峰值 lr `3e-4 → 1.5e-4`，前 2 个 epoch 线性 warmup |
| ③ | **梯度裁剪** | 防止单批梯度发散的廉价保险 | `clip_grad_norm_(1.0)` |
| ④ | **更强的针对性数据增强** | 提升跨司机泛化 | RandAugment + RandomErasing，放宽 crop scale |
| ⑥ | **延长训练到 24 epoch** | 稳定+正则后让训练充分收敛 | `epochs: 24`，cosine 衰减跟随 |

> ⚠️ **绝不能加水平翻转**：类别区分左右手动作（`texting_left` vs `texting_right` 等），翻转会让标签错乱。torchvision 的 RandAugment 操作空间不含翻转，因此可安全使用。

未采用（备选，效果不够再上）：
- ⑤ 权重 EMA（需改训练循环，本次为控制改动范围未做）
- 提高 `weight_decay`（5e-4 → 1e-2）
- 对 `safe_driving` / `talking_to_passenger` 加 `class_weights` 缓解混淆

---

## 3. 代码改动

| 文件 | 改动 | 对应策略 |
| --- | --- | --- |
| `driver_distraction/models/factory.py` | `build_mobilenet_v4` / `build_model` 新增 `drop_path_rate` 参数 | ① |
| `scripts/train.py` | 从 `config["train"]["mobilenet_v4_drop_path"]` 透传 `drop_path_rate` | ① |
| `driver_distraction/engine/trainer.py` | `build_scheduler` 支持 `warmup_epochs`（LinearLR + Cosine 用 SequentialLR 串联）；`train_one_epoch` 增加 `grad_clip_norm`（AMP 下先 `unscale_` 再裁剪） | ②③ |
| `driver_distraction/data/transforms.py` | `build_train_transform` 增加 RandAugment、RandomErasing（均可配置、默认开），放宽 RandomResizedCrop scale | ④ |
| `configs/config.yaml` | 见下方超参对照 | ②③④⑥ |
| `requirements.txt` | 新增 `timm>=1.0.0` | （MobileNetV4 依赖） |

### 超参对照（`configs/config.yaml`）

| 项 | 改前 | 改后 |
| --- | --- | --- |
| `train.lr` | 0.0003 | **0.00015** |
| `train.epochs` | 12 | **24** |
| `train.warmup_epochs` | （无） | **2** |
| `train.grad_clip_norm` | （无） | **1.0** |
| `train.mobilenet_v4_drop_path` | （无） | **0.1** |
| `augmentation.random_resized_crop_scale` | [0.85, 1.0] | **[0.6, 1.0]** |
| `augmentation.randaugment` | （无） | **enabled, num_ops=2, magnitude=7** |
| `augmentation.random_erasing` | （无） | **enabled, p=0.25** |

> 说明：`lr / epochs / 增强` 是全局 `train` 默认值，调整后也会影响重新训练的 V3/ResNet；`mobilenet_v4_drop_path` 仅 MobileNetV4 分支读取，其他模型忽略。已训练好的 V3 checkpoint 不受影响。

### 训练增强链（改后）

```text
Resize → RandomResizedCrop → ColorJitter → RandomRotation → RandAugment
       → ToTensor → Normalize → RandomErasing
```

### 学习率曲线（验证输出）

```text
warmup: 1.5e-5 → 8.25e-5 → 1.5e-4（第 3 轮到峰值）
cosine: 1.5e-4 → … → 8e-7（第 24 轮）
```

---

## 4. 训练与评估命令

首次需安装 timm：

```powershell
D:\Anaconda_envs\envs\driver_ai\python.exe -m pip install "timm>=1.0.0"
```

清掉上一次 v4 旧输出，避免 `history.csv` 把新旧两次 run 混在一起：

```powershell
Remove-Item -Recurse -Force outputs\checkpoints\mobilenet_v4 -ErrorAction SilentlyContinue
```

训练（超参已写入 config，命令不变）：

```powershell
D:\Anaconda_envs\envs\driver_ai\python.exe scripts\train.py --config configs\config.yaml --model mobilenet_v4 --device cuda
```

评估：

```powershell
D:\Anaconda_envs\envs\driver_ai\python.exe scripts\evaluate.py --config configs\config.yaml --model mobilenet_v4 --checkpoint outputs\checkpoints\mobilenet_v4\best.pt --split test --batch-size 64 --num-workers 4 --output-dir outputs\reports\mobilenet_v4\test
```

---

## 5. 如何判断策略是否生效

跑完后对比 `outputs/checkpoints/mobilenet_v4/history.csv`：

1. **`val_loss` 不再爆炸** —— 不应再出现 2.7 / 7.2 这类尖峰（warmup + 梯度裁剪的作用）。
2. **train/val 差距收窄** —— `train_acc` 应被增强 + drop_path 压到约 0.90，而 val 反而更高。
3. **`best_epoch` 后移** —— 不再停在第 2 轮，理想落在 12~20 轮之间，说明 24 轮没有空跑。

---

## 6. EMA优化

 What I changed

  New driver_distraction/utils/ema.py — a ModelEMA class holding a shadow copy of the weights, updated each step as ema
  = decay·ema + (1−decay)·model. Non-float buffers (e.g. BN num_batches_tracked) are copied verbatim, and decay ramps up
  from ~0 during early steps (timm-style warmup) so the average isn't polluted by near-random init weights.

  driver_distraction/engine/trainer.py
  - build_ema() constructs the tracker from config (returns None when disabled).
  - train_one_epoch() calls ema.update(model) after every optimizer.step().
  - fit() evaluates the EMA model each epoch — those metrics drive best-checkpoint selection.
  - Checkpointing: last.pt keeps raw weights (exact resume) plus an ema_state; best.pt's model_state is written with the
  EMA weights. Since every downstream consumer (evaluate.py, export_onnx.py, grad_cam.py, camera_demo.py, finetune)
  already reads model_state from best.pt, they all automatically deploy the smoother EMA model — no changes needed
  there.
  - Resume restores the EMA shadow and its update counter.
  - Per-epoch log lines tag val metrics with ema when active.

  configs/config.yaml — added a train.ema block (enabled: true, decay: 0.999, warmup: true).

  Notes

  - I chose decay: 0.999 rather than the ImageNet-typical 0.9998. For this ~24-epoch finetune (~470 steps/epoch ≈ 11k
  updates), 0.999 gives a ~1k-step averaging window (~2 epochs) — long enough to smooth, short enough to keep up. Bump
  it toward 0.9997–0.9998 if you later train much longer.
  - Backward compatible: set ema.enabled: false and behavior reverts exactly to before (raw weights evaluated and
  saved).
  - Verified with a unit smoke test (update/lag/state round-trip/warmup ramp) and a full 2-epoch fit() run confirming
  best.pt holds EMA weights, last.pt holds raw weights, and resume restores the EMA state.

  To use it, just train as usual (python scripts/train.py); to resume an existing run with EMA now enabled, the tracker
  initializes from the current weights and warms up from there.
