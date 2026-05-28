# Splits

Place the four fixed split files provided by the TA here:

```
splits/
├── train_ssl_unlabeled.txt
├── train_labeled_10percent.txt
├── val.txt
└── test.txt
```

Each file should contain one integer index per line. Indices in
`train_ssl_unlabeled.txt`, `train_labeled_10percent.txt`, and `val.txt`
refer to the **CIFAR-10 train set**; `test.txt` refers to the
**CIFAR-10 test set**.

Do **not** modify these files or create your own splits.
