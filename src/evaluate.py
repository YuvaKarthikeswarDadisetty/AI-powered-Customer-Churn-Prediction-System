from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    return y_pred

def plot_confusion_matrix(y_test, y_pred):
    os.makedirs("outputs", exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")

    plt.savefig("outputs/confusion_matrix.png")
    print("📊 Confusion matrix saved")
    plt.close()

def plot_roc_curve(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]

    # ✅ FIX: Explicit pos_label
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.title("ROC Curve")
    plt.legend()

    plt.savefig("outputs/roc_curve.png")
    print("📊 ROC curve saved")
    plt.close()