# Évaluation du modèle
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    return {
        "F1 Score": f1,
        "AUC-ROC": auc,
        "Confusion Matrix": matrix
    }
