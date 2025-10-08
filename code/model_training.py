# Entraînement du modèle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(X, y):
    # Division des données
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialisation du modèle
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    return model, X_val, y_val, X_test, y_test
