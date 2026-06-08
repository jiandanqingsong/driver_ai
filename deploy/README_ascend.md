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

## 3. Deployment placeholders

After OM generation, connect the `.om` model with one of these inference runtimes:

- Ascend ACL C++/Python
- MindX SDK StreamManager
- Atlas edge device application

Keep preprocessing consistent with `driver_distraction/data/transforms.py`.
