from sklearn.metrics import f1_score, recall_score


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def macro_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average="macro", zero_division=0)
