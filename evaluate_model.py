from sklearn.metrics import f1_score


def evaluate_model(y_pred, y_true):
    return f1_score(y_pred, y_true)
