import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

try:
    # When executed as a module: python -m app.models.src.data.eda.preproces.train
    from . import explain  # type: ignore
    from . import preprocess_data as prep  # type: ignore
except Exception:  # Fallback when run directly from this folder
    import explain  # type: ignore
    import preprocess_data as prep  # type: ignore


def train_and_save_model() -> str:
    df, _ = explain.load_data()
    X, y = prep.split_features_and_target(df, target_column="target")
    X_train, X_test, y_train, y_test, scaler = prep.train_test_split_scaled(X, y)

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    prep.ensure_dir(artifacts_dir)
    model_path = os.path.join(artifacts_dir, "cancer_model.pkl")
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")
    return model_path


if __name__ == "__main__":
    train_and_save_model()


