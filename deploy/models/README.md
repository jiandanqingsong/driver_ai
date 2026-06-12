# Deployment Models

Put board deployment model files here.

Recommended file:

```text
mobilenet_v3_large_demo_finetune_driver_distraction.onnx
mobilenet_v4_ema_demo_finetune_balanced_driver_distraction.onnx
```

The generated `.om` file is not tracked by Git. Convert it on the Ascend board with:

```bash
bash deploy/ascend_atc_template.sh
```

Convert the MobileNetV4 EMA balanced model:

```bash
bash deploy/convert_mobilenet_v4_ema_balanced_to_om.sh
```

The default output is:

```text
deploy/models/mobilenet_v4_ema_demo_finetune_balanced_driver_distraction.om
```

Run one-image inference after conversion:

```bash
python3 scripts/ascend_infer.py \
  --model deploy/models/mobilenet_v3_large_demo_finetune_driver_distraction.om \
  --image data/test_driver.jpg
```
