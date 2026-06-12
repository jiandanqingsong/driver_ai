# Ascend OM Export Notes

This project exports PyTorch checkpoints to ONNX first, then converts ONNX to OM by Ascend ATC.

## 1. Export ONNX

```bash
python scripts/export_onnx.py --config configs/config.yaml
```

## 2. Convert ONNX to OM

Run this step on a machine with Ascend Toolkit installed:

```bash
bash deploy/ascend_atc_template.sh
```

Or generate the command in Python:

```python
from driver_distraction.deploy.ascend import build_atc_command
from driver_distraction.utils.config import load_config

cfg = load_config("configs/config.yaml")
print(" ".join(build_atc_command(cfg)))
```

## 3. Run ACL Python inference

After OM generation, install the board-side dependencies:

```bash
python3 -m pip install -r deploy/requirements_board.txt
```

Then run:

```bash
python3 scripts/ascend_infer.py \
  --model deploy/models/mobilenet_v3_large_demo_finetune_driver_distraction.om \
  --image data/test_driver.jpg
```

The implementation is in `driver_distraction/deploy/acl_infer.py`. It uses
AscendCL Python directly and keeps preprocessing consistent with
`driver_distraction/data/transforms.py`.

See `deploy/README_board.md` for the complete board workflow.
