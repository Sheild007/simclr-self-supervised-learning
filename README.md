# DL Assignment 5 — SimCLR (BSCS22008)

## Run

```bash
pip install -r requirements.txt
```

Place the four split files in `splits/`:

```
splits/train_ssl_unlabeled.txt
splits/train_labeled_10percent.txt
splits/val.txt
splits/test.txt
```

Run everything in order:

```bash
python BSCS22008_05_allCode.py
```

Or run one task at a time:

```bash
python BSCS22008_05_task1_supervised.py
python BSCS22008_05_task2_augmentations.py
python BSCS22008_05_task3_similarity.py
python BSCS22008_05_task4_simclr.py
python BSCS22008_05_task5_linear_probe.py
python BSCS22008_05_task6_finetune.py
```
