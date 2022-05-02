import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.where(pred.predictions > 0.5, 1.0, 0.0)

    label_indices = list(range(2))
    f1 = f1_score(labels, preds, average="micro", labels=label_indices) * 100.0
    acc = accuracy_score(labels, preds)
    return {
        'micro f1 score': f1,
        'accuracy': acc,
    }
