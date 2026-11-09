from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)
import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def _scale_features(train, test):
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    std[std == 0] = 1.0
    return (train - mean) / std, (test - mean) / std


def _get_metrics(y_true, y_pred, y_prob) -> ClassificationMetrics:
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X_train = np.array(data.train_features, dtype=float)
    y_train = np.array(data.train_target, dtype=float)
    X_test = np.array(data.test_features, dtype=float)

    X_train_scaled, X_test_scaled = _scale_features(X_train, X_test)

    n_samples, n_features = X_train_scaled.shape
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []

    lr = data.learning_rate
    epochs = data.epochs

    for _ in range(epochs):
        y_pred = X_train_scaled @ weights + bias
        error = y_pred - y_train
        mse = np.mean(error ** 2)
        loss_history.append(float(mse))

        dw = (2 / n_samples) * (X_train_scaled.T @ error)
        db = (2 / n_samples) * np.sum(error)

        weights -= lr * dw
        bias -= lr * db

    y_pred_test = X_test_scaled @ weights + bias

    return RegressionResult(
        weights=weights.tolist(),
        bias=float(bias),
        loss_history=loss_history,
        predictions=y_pred_test.tolist(),
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    X_train = np.array(data.train_features, dtype=float)
    y_train = np.array(data.train_target, dtype=float)
    X_test = np.array(data.test_features, dtype=float)

    X_train_scaled, X_test_scaled = _scale_features(X_train, X_test)

    n_samples, n_features = X_train_scaled.shape
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []

    lr = data.learning_rate
    epochs = data.epochs

    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    for _ in range(epochs):
        z = X_train_scaled @ weights + bias
        y_pred = sigmoid(z)

        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        bce = -np.mean(y_train * np.log(y_pred_clipped) + (1 - y_train) * np.log(1 - y_pred_clipped))
        loss_history.append(float(bce))

        error = y_pred - y_train
        dw = (1 / n_samples) * (X_train_scaled.T @ error)
        db = (1 / n_samples) * np.sum(error)

        weights -= lr * dw
        bias -= lr * db

    y_pred_proba = sigmoid(X_test_scaled @ weights + bias)
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    return ClassificationResult(
        weights=weights.tolist(),
        bias=float(bias),
        loss_history=loss_history,
        probabilities=y_pred_proba.tolist(),
        predictions=y_pred_binary.tolist(),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    X_train = np.array(data.train_features, dtype=float)
    y_train = np.array(data.train_target, dtype=float)
    X_test = np.array(data.test_features, dtype=float)
    y_test = np.array(data.test_target, dtype=float)

    sk_log = SklearnLogReg(random_state=42, max_iter=1000)
    sk_log.fit(X_train, y_train)
    sk_preds = sk_log.predict(X_test)
    sk_proba = sk_log.predict_proba(X_test)[:, 1]

    dt = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    dt_proba = dt.predict_proba(X_test)[:, 1]

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    knn_proba = knn.predict_proba(X_test)[:, 1]

    return BaselineComparisonResult(
        logistic_regression=_get_metrics(y_test, sk_preds, sk_proba),
        decision_tree=_get_metrics(y_test, dt_preds, dt_proba),
        knn=_get_metrics(y_test, knn_preds, knn_proba)
    )
