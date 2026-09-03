# Model comparison — Phase 1

Test split: `test` (14932 face crops), decision threshold 0.5.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Infer ms/img |
|---|---|---|---|---|---|---|
| efficientnet_b0 | 0.9912 | 0.9887 | 0.9938 | 0.9912 | 0.9989 | 2.76 |
| xception | 0.9858 | 0.9919 | 0.9796 | 0.9857 | 0.9990 | 2.08 |

## Conclusion

**xception** achieves the highest ROC-AUC (0.9990); **efficientnet_b0** has the best F1 (0.9912), and **xception** is the fastest at 2.08 ms/image on NVIDIA GeForce RTX 4060 Laptop GPU. The ROC-AUC spread across the models is 0.0001, below the 0.001 margin this report treats as a tie, so the decision falls to F1 at the 0.5 threshold: **efficientnet_b0** is the checkpoint the backend should load.

## Run details

- **efficientnet_b0** — checkpoint `machine-learning/checkpoints/efficientnet_b0_20260901_204509`, best epoch 8, 14932 test crops, confusion matrix {'tn': 7388, 'fp': 85, 'fn': 46, 'tp': 7413}
- **xception** — checkpoint `machine-learning/checkpoints/xception_20260901_210926`, best epoch 6, 14932 test crops, confusion matrix {'tn': 7413, 'fp': 60, 'fn': 152, 'tp': 7307}
