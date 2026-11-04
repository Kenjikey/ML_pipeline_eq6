from pathlib import Path
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def entrenar_modelo_svm_entradas():
    # 1. Conexión con MLflow local (Puerto 5001)
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("mibici_entrenamiento_svm")

    # 2. Carga de los datos procesados (33 columnas)
    ruta_processed = Path("data/processed")
    archivos_parquet = list(ruta_processed.glob("*.parquet"))

    if not archivos_parquet:
        raise FileNotFoundError(
            "No se encontraron archivos .parquet en data/processed/"
        )

    print(f"Cargando datos desde: {archivos_parquet[0]}")
    df = pd.read_parquet(archivos_parquet[0])

    # 3. Separar variables explicativas (X) y objetivo (y = entradas)
    columnas_excluir = ["timestamp", "entradas", "salidas", "balance"]
    feature_cols = [col for col in df.columns if col not in columnas_excluir]

    X = df[feature_cols]
    y = df["entradas"]

    # División de datos temporal (Train / Validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # Escalado estándar (Requisito fundamental para optimización en SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 4. Registro y ejecución en MLflow
    with mlflow.start_run(run_name="svm_entradas_10_epocas"):
        epochs = 10

        # Modelo SVM lineal iterativo (SGDRegressor con epsilon_insensitive)
        model = SGDRegressor(
            loss="epsilon_insensitive",
            epsilon=0.1,
            learning_rate="optimal",
            random_state=42,
        )

        # Registrar parámetros iniciales
        mlflow.log_params(
            {
                "model_type": "Support_Vector_Machine_LinearSVR",
                "target_variable": "entradas",
                "epochs": epochs,
                "n_features": len(feature_cols),
                "scaler": "StandardScaler",
            }
        )

        print(
            f"\nIniciando entrenamiento SVM para 'entradas' ({epochs} épocas)..."
        )

        # Bucle de entrenamiento por épocas
        for epoch in range(1, epochs + 1):
            # Entrenar la época actual
            model.partial_fit(X_train_scaled, y_train)

            # Evaluaciones de validación
            y_pred_val = model.predict(X_val_scaled)
            val_mae = mean_absolute_error(y_val, y_pred_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            val_r2 = r2_score(y_val, y_pred_val)

            # Registrar métricas en MLflow asociadas a la época actual
            mlflow.log_metric("val_mae", val_mae, step=epoch)
            mlflow.log_metric("val_rmse", val_rmse, step=epoch)
            mlflow.log_metric(
                "val_r2_accuracy", val_r2, step=epoch
            )  # R2 / Accuracy

            print(
                f"Época {epoch:02d}/{epochs:02d} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | R2 (Accuracy): {val_r2:.4f}"
            )

        # Registrar artefacto del modelo entrenado
        mlflow.sklearn.log_model(model, artifact_path="modelo_svm_entradas")
        print(
            "\n¡Entrenamiento completado y registrado exitosamente en MLflow!"
        )


if __name__ == "__main__":
    entrenar_modelo_svm_entradas()