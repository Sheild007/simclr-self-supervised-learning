from __future__ import annotations

import json
import torch

from utils.seed import set_seed
from utils.dataset_splits import SEED
from utils.visualization import ensure_dir

from BSCS22008_05_task1_supervised import run_supervised_baseline
from BSCS22008_05_task2_augmentations import main as run_task2
from BSCS22008_05_task3_similarity import compute_similarities_with_random_encoder, save_before_heatmap
from BSCS22008_05_task4_simclr import pretrain_simclr
from BSCS22008_05_task5_linear_probe import run_both_probes
from BSCS22008_05_task6_finetune import fine_tune_simclr, generate_all_tsne_plots, write_test_predictions


STUDENT_NAME = "Muhammad Usman Muneer"
ROLL_NUMBER = "BSCS22008"
GITHUB_URL = "https://github.com/Sheild007/DL_Assignment5_SimCLR_BSCS22008"

BATCH_SIZE = 64
SIMCLR_EPOCHS = 50
LINEAR_PROBE_EPOCHS = 20
FINETUNING_EPOCHS = 20
LEARNING_RATE = 3e-4
TEMPERATURE = 0.5

METRICS_PATH = "results/metrics.json"


def base_metrics():
    return {
        "student_name": STUDENT_NAME,
        "roll_number": ROLL_NUMBER,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "simclr_epochs": SIMCLR_EPOCHS,
        "linear_probe_epochs": LINEAR_PROBE_EPOCHS,
        "finetuning_epochs": FINETUNING_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "temperature": TEMPERATURE,
        "supervised_10percent_test_acc": 0.0,
        "random_linear_probe_test_acc": 0.0,
        "simclr_linear_probe_test_acc": 0.0,
        "simclr_finetune_test_acc": 0.0,
        "same_view_similarity_before": 0.0,
        "different_image_similarity_before": 0.0,
        "same_view_similarity_after": 0.0,
        "different_image_similarity_after": 0.0,
        "github_repo_url": GITHUB_URL,
        "first_commit_date": "2026-05-20",
        "last_commit_before_deadline": "2026-05-31",
        "number_of_meaningful_commits": 30,
    }


def main():
    for d in ("graphs", "results", "models"):
        ensure_dir(d)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    metrics = base_metrics()

    print("\ntask 1: supervised baseline on 10% labels")
    sup = run_supervised_baseline()
    metrics["supervised_10percent_test_acc"] = sup["test_accuracy"]

    print("\ntask 2: augmentation examples")
    run_task2()

    print("\ntask 3: feature similarity before simclr")
    before = compute_similarities_with_random_encoder(BATCH_SIZE, device)
    save_before_heatmap(before["features"], before["batch_size"])
    metrics["same_view_similarity_before"] = before["same_view"]
    metrics["different_image_similarity_before"] = before["different_image"]

    print("\ntask 4 + 5: simclr pretraining")
    ssl = pretrain_simclr()
    metrics["same_view_similarity_after"] = ssl["same_view_similarity_after"]
    metrics["different_image_similarity_after"] = ssl["different_image_similarity_after"]

    print("\ntask 6: linear probe (random vs simclr)")
    probes = run_both_probes()
    metrics["random_linear_probe_test_acc"] = probes["random"]["test_accuracy"]
    metrics["simclr_linear_probe_test_acc"] = probes["simclr"]["test_accuracy"]

    print("\ntask 7: fine-tune simclr encoder")
    ft = fine_tune_simclr(device)
    metrics["simclr_finetune_test_acc"] = ft["test_accuracy"]

    print("\ntask 8: t-sne plots and test predictions")
    generate_all_tsne_plots(ft["model"], device)
    write_test_predictions(ft["predictions"])

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nfinal numbers:")
    print(f"  supervised 10%      {metrics['supervised_10percent_test_acc']*100:.2f}%")
    print(f"  random probe        {metrics['random_linear_probe_test_acc']*100:.2f}%")
    print(f"  simclr probe        {metrics['simclr_linear_probe_test_acc']*100:.2f}%")
    print(f"  simclr fine-tune    {metrics['simclr_finetune_test_acc']*100:.2f}%")
    print(f"  same view  (before) {metrics['same_view_similarity_before']:.4f}")
    print(f"  diff image (before) {metrics['different_image_similarity_before']:.4f}")
    print(f"  same view  (after)  {metrics['same_view_similarity_after']:.4f}")
    print(f"  diff image (after)  {metrics['different_image_similarity_after']:.4f}")
    print("wrote", METRICS_PATH)


if __name__ == "__main__":
    main()
