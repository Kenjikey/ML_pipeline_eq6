from pathlib import Path
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def entrenar_sgd_con_epocas(epochs: int = 10):
    # 1. Configurar servidor de MLflow local (Puerto 5001)
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("mibici_sgd_epocas")

    # 2. Cargar dataset procesado
    ruta_processed = Path("data/processed")
    archivos_parquet = list(ruta_processed.glob("*.parquet"))

    if not archivos_parquet:
        raise FileNotFoundError(
            "No se encontraron archivos .parquet en data/processed/"
        )

    print(f"Cargando dataset desde: {archivos_parquet[0]}")
    df = pd.read_parquet(archivos_parquet[0])

    # 3. Separar características (X) y objetivo (y = entradas)
    columnas_excluir = ["timestamp", "entradas", "salidas", "balance"]
    feature_cols = [col for col in df.columns if col not in columnas_excluir]

    X = df[feature_cols]
    y = df["entradas"]

    # División cronológica (80% Train, 20% Val)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # Escalado estándar (requerido para optimización SGD)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 4. Inicializar ejecución en MLflow
    with mlflow.start_run(run_name=f"sgd_regressor_{epochs}_epocas"):
        model = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=0.0001,
            learning_rate="optimal",
            random_state=42,
        )

        mlflow.log_params(
            {
                "model_type": "SGDRegressor",
                "target_variable": "entradas",
                "epochs": epochs,
                "loss": "squared_error",
                "learning_rate": "optimal",
                "scaler": "StandardScaler",
            }
        )

        print(
            f"\nIniciando entrenamiento iterativo con SGDRegressor ({epochs} épocas)..."
        )

        # Bucle por épocas
        for epoch in range(1, epochs + 1):
            model.partial_fit(X_train_scaled, y_train)

            y_pred_val = model.predict(X_val_scaled)

            mae = mean_absolute_error(y_val, y_pred_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            r2 = r2_score(y_val, y_pred_val)

            # Registro con step=epoch para la gráfica en MLflow
            mlflow.log_metric("val_mae", mae, step=epoch)
            mlflow.log_metric("val_rmse", rmse, step=epoch)
            mlflow.log_metric("val_r2", r2, step=epoch)

            print(
                f"Época {epoch:02d}/{epochs:02d} | Val MAE: {mae:.4f} | Val RMSE: {rmse:.4f} | Val R2: {r2:.4f}"
            )

        mlflow.sklearn.log_model(model, name="modelo_sgd_entradas")
        print(
            "\n¡Entrenamiento finalizado y las 10 épocas registradas en MLflow!"
        )


if __name__ == "__main__":
    entrenar_sgd_con_epocas(epochs=10)