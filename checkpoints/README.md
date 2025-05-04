
Hello! This directory contains trained model checkpoints, evaluation results, and analysis files for various UNet models trained on biodiversity datasets.


Each subdirectory represents a different model training run with naming convention:
```
[DATASET][LR][WEIGHT][BATCH][EPOCHS]
```

For example:
- `ASAL1e-07W1e-08B2E15`: Dataset A with self-annotation, LR=1e-7, Weight Decay=1e-8, Batch Size=2, 15 Epochs

- `BL1e-06W1e-07B4E10`: Dataset B, LR=1e-6, Weight Decay=1e-7, Batch Size=4, 10 Epochs



Each model directory typically contains

- `output.txt`: Training logs including loss and validation metrics
- `metrics_comparison*.txt`: Analysis of differences between Overall Accuracy (OA) and F1 Score

`/results/`: Directory containing evaluation results:
  - `confusion_matrix.png`: Visualization of class prediction confusion matrix
  - `class_accuracies.png`: Bar chart of per-class accuracy
  - `evaluation_report.txt`: Detailed performance metrics

The actual .pth files will not be uploaded due to size constraints but they can be made available upon request.

## Summary Files


- `validation_scores.json`: Dictionary of validation F1 scores for all models
- `validation_scores_sorted.txt`: Models ranked by validation F1 score
- `overall_accuracies.json`: Dictionary of overall accuracy for all models
- `overall_accuracies_sorted.txt`: Models ranked by overall accuracy
- `metrics_differences.json`: Dictionary of differences between Overall Accuracy and F1 score
- `optimal_runs.txt`: List of models meeting optimal criteria (small OA-F1 difference, good class coverage)


The following utility scripts are available to help analyze results:

 _sort_validation_json.py_: Sorts and formats validation scores

_sort_overall_accuracies.py_: Sorts and formats overall accuracy scores
