# Deployment Models

Put board deployment model files here.

Recommended file:

```text
mobilenet_v3_large_demo_finetune_driver_distraction.onnx
```

The generated `.om` file is not tracked by Git. Convert it on the Ascend board with:

```bash
bash deploy/ascend_atc_template.sh
```

Run one-image inference after conversion:

```bash
python3 scripts/ascend_infer.py \
  --model deploy/models/mobilenet_v3_large_demo_finetune_driver_distraction.om \
  --image data/test_driver.jpg
```
