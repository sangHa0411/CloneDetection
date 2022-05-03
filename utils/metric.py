import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    similar_flag = True if len(pred.predictions) == 2 else False 

    if similar_flag :
        preds = np.where(pred.predictions >= 0.5, 1, 0)
    else :
        preds = np.argmax(pred.predictions, axis=-1)

    label_indices = list(range(2))
    f1 = f1_score(labels, preds, average="micro", labels=label_indices) * 100.0
    acc = accuracy_score(labels, preds)
    return {
        'micro f1 score': f1,
        'accuracy': acc,
    }
