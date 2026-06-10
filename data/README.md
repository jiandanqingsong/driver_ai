# Data Directory

Place the Kaggle State Farm Distracted Driver Detection dataset here.

Expected layout:

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

The Kaggle metadata file `driver_imgs_list.csv` is required for driver-id split.
D:\Anaconda_envs\envs\driver_ai\python.exe scripts\evaluate.py --config configs\config.yaml --model mobilenet_v4 --checkpoint outputs\checkpoints\mobilenet_v4\best.pt --split test --batch-size 64 --num-workers 4 --output-dir outputs\reports\mobilenet_v4\test