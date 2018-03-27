from sklearn.metrics import roc_auc_score


def auc(model, X, y):
    yhat = model.predict_proba(X)
    return roc_auc_score(y, yhat)

class Evaluator:

    def __init__(self, X, y,
                 eval_fn = auc,
                 eval_metric_name = "auc"):
        """
        X: a matrix where rows are features and columns are examples
        y: the true labels for the columns of X
        eval_fn: a function that takes a model and returns an accuracy comparison
        of comparing y (the true labels) to the result of predicting on X
        with the passed model.
        """
        self.X = X
        self.y = y
        self.eval_fn = eval_fn
        self.eval_metric_name = eval_metric_name

    def evaluate(self, model):
        return self.eval_fn(model, self.X, self.y)
    