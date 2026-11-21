import json
from pathlib import Path
import subprocess

import joblib
from loguru import logger
import mlflow
import mlflow.data
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import typer

from mibici.config import (
    FEATURES_FILENAME,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    REGISTERED_MODEL_NAME,
)
from mibici.integrity import verify_dataset_manifest
from mibici.modeling.evaluate import (
    BaselineHeuristicModel,
    calculate_metrics,
    evaluate_net_balance,
)
from mibici.modeling.registry import register_model
from mibici.pipeline import (
    ALL_FEATURE_COLS,
    TARGET_COLS,
    create_full_pipeline,
    get_feature_groups,
)

app = typer.Typer()


def split_out_of_time(df: pd.DataFrame):
    """Particion cronologica estricta: Train (Ene-Sep), Val (Oct), Test (Nov-Dic)."""
    train_df = df[df["mes"] <= 9].copy()
    val_df = df[df["mes"] == 10].copy()
    test_df = df[df["mes"] >= 11].copy()

    logger.info(
        f"Split temporal -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}"
    )
    return train_df, val_df, test_df


def evaluate_baselines(
    train_df: pd.DataFrame, eval_df: pd.DataFrame, prefix: str = "test"
) -> dict:
    """Calcula metricas de Baseline 0 y Baseline Heuristico (mediana) en val o test."""
    y_true_in = eval_df["entradas"].values
    y_true_out = eval_df["salidas"].values
    y_true_balance = eval_df["balance"].values

    mae_base0 = float(mean_absolute_error(y_true_balance, np.zeros_like(y_true_balance)))
    rmse_base0 = float(root_mean_squared_error(y_true_balance, np.zeros_like(y_true_balance)))

    heur_in = BaselineHeuristicModel().fit(train_df, target_col="entradas")
    heur_out = BaselineHeuristicModel().fit(train_df, target_col="salidas")

    pred_heur_in = heur_in.predict(eval_df)
    pred_heur_out = heur_out.predict(eval_df)
    bal_heur = evaluate_net_balance(y_true_in, pred_heur_in, y_true_out, pred_heur_out)

    return {
        f"{prefix}_mae_base0": round(mae_base0, 4),
        f"{prefix}_rmse_base0": round(rmse_base0, 4),
        f"{prefix}_mae_heur": bal_heur["balance_mae"],
        f"{prefix}_rmse_heur": bal_heur["balance_rmse"],
        f"{prefix}_r2_heur": bal_heur["balance_r2"],
    }


def select_by_validation(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    candidates: list[dict],
):
    """Entrena candidatos en Train y selecciona el ganador por menor val_balance_mae."""
    best_pipeline = None
    best_params = None
    best_val_metrics = None
    best_score = float("inf")

    for idx, candidate in enumerate(candidates, start=1):
        est_params = {k: v for k, v in candidate.items() if k != "model_family"}
        pipe = create_full_pipeline(est_params)
        pipe.fit(X_train, y_train)

        val_preds = np.clip(pipe.predict(X_val), 0, None)
        if not np.isfinite(val_preds).all():
            raise ValueError(f"Predicciones no finitas (NaN/inf) en candidato {idx}.")

        val_metrics = {
            **calculate_metrics(y_val["entradas"].values, val_preds[:, 0], prefix="val_ent"),
            **calculate_metrics(y_val["salidas"].values, val_preds[:, 1], prefix="val_sal"),
            **{
                f"val_{k}": v
                for k, v in evaluate_net_balance(
                    y_val["entradas"].values,
                    val_preds[:, 0],
                    y_val["salidas"].values,
                    val_preds[:, 1],
                ).items()
            },
        }
        score = val_metrics["val_balance_mae"]
        logger.info(
            f"Candidato {idx}/{len(candidates)} ({est_params}) -> val_balance_mae={score:.4f}"
        )

        if score < best_score:
            best_score = score
            best_pipeline = pipe
            best_params = candidate
            best_val_metrics = val_metrics

    return best_pipeline, best_params, best_val_metrics


def _git_commit() -> str:
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, capture_output=True, text=True
    )
    return res.stdout.strip() if res.returncode == 0 else "no_disponible"


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / FEATURES_FILENAME,
    models_dir: Path = MODELS_DIR,
    max_iter: int = 150,
    max_depth: int = 10,
    learning_rate: float = 0.08,
    random_state: int = 42,
):
    manifest_info = verify_dataset_manifest(features_path, manifest_id="features_2023_v2")
    logger.info(f"Integridad verificada ({manifest_info['id']}): SHA-256 OK")

    df = pd.read_parquet(features_path)
    missing = set(ALL_FEATURE_COLS + TARGET_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing}")

    train_df, val_df, test_df = split_out_of_time(df)
    X_train, y_train = train_df[ALL_FEATURE_COLS], train_df[TARGET_COLS]
    X_val, y_val = val_df[ALL_FEATURE_COLS], val_df[TARGET_COLS]
    X_test, y_test = test_df[ALL_FEATURE_COLS], test_df[TARGET_COLS]

    val_baseline_metrics = evaluate_baselines(train_df, val_df, prefix="val")
    test_baseline_metrics = evaluate_baselines(train_df, test_df, prefix="test")
    baseline_metrics = {**val_baseline_metrics, **test_baseline_metrics}

    candidates = [
        {
            "loss": "poisson",
            "max_iter": 450,
            "max_depth": None,
            "max_leaf_nodes": 127,
            "min_samples_leaf": 30,
            "learning_rate": 0.05,
            "l2_regularization": 0.1,
            "random_state": random_state,
        },
        {
            "loss": "poisson",
            "max_iter": 500,
            "max_depth": None,
            "max_leaf_nodes": 255,
            "min_samples_leaf": 50,
            "learning_rate": 0.04,
            "l2_regularization": 0.5,
            "random_state": random_state,
        },
    ]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="hgb_multioutput_pipeline_v2") as run:
        commit_hash = _git_commit()
        mlflow.set_tags(
            {
                "stage": "train",
                "dataset_id": manifest_info["id"],
                "dataset_sha256": manifest_info["sha256"],
                "git_commit": commit_hash,
                "team_phase": "final",
            }
        )
        mlflow.log_metrics(baseline_metrics)

        dataset_source = mlflow.data.from_pandas(
            df=train_df[ALL_FEATURE_COLS + TARGET_COLS],
            source=str(features_path.resolve()),
            name=manifest_info["id"],
        )
        mlflow.log_input(dataset_source, context="training")

        logger.info("Seleccionando mejor configuracion usando Validacion (Octubre)...")
        pipeline, best_params, val_metrics = select_by_validation(
            X_train, y_train, X_val, y_val, candidates
        )

        # Compuerta de calidad antes de evaluar Test o registrar modelo
        if val_metrics["val_balance_mae"] >= baseline_metrics["val_mae_heur"]:
            raise ValueError(
                f"El modelo ({val_metrics['val_balance_mae']}) no supera al baseline en "
                f"validacion ({baseline_metrics['val_mae_heur']}). Registro abortado."
            )

        mlflow.log_params(best_params)
        mlflow.log_param("n_features", len(ALL_FEATURE_COLS))

        # Evaluacion unica en Test (Noviembre - Diciembre)
        test_preds = np.clip(pipeline.predict(X_test), 0, None)
        test_bal = evaluate_net_balance(
            y_test["entradas"].values,
            test_preds[:, 0],
            y_test["salidas"].values,
            test_preds[:, 1],
        )
        mejora_heur = (
            (baseline_metrics["test_mae_heur"] - test_bal["balance_mae"])
            / baseline_metrics["test_mae_heur"]
        ) * 100

        test_metrics = {
            **calculate_metrics(y_test["entradas"].values, test_preds[:, 0], prefix="test_ent"),
            **calculate_metrics(y_test["salidas"].values, test_preds[:, 1], prefix="test_sal"),
            **{f"test_{k}": v for k, v in test_bal.items()},
            "test_mejora_vs_heuristico_pct": round(mejora_heur, 2),
        }

        mlflow.log_metrics({**val_metrics, **test_metrics})
        logger.info(f"Ganador Validacion (Oct): {val_metrics}")
        logger.info(f"Evaluacion Test (Nov-Dic): {test_metrics}")

        models_dir.mkdir(parents=True, exist_ok=True)
        pipeline_path = models_dir / "mibici_balance_pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)

        schema_payload = {
            "model_name": REGISTERED_MODEL_NAME,
            "run_id": run.info.run_id,
            "dataset_id": manifest_info["id"],
            "dataset_sha256": manifest_info["sha256"],
            "git_commit": commit_hash,
            "feature_groups": get_feature_groups(),
            "dtypes": {col: str(X_train[col].dtype) for col in ALL_FEATURE_COLS},
            "hyperparameters": best_params,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        schema_path = models_dir / "model_schema.json"
        schema_path.write_text(json.dumps(schema_payload, indent=2) + "\n", encoding="utf-8")

        mlflow.log_artifact(str(schema_path), artifact_path="metadata")
        mlflow.log_artifact(str(pipeline_path), artifact_path="serialized_pipeline")

        # Registro versionado y etiquetado en MLflow Model Registry
        sample_X = X_train.head(5)
        signature = infer_signature(sample_X, pipeline.predict(sample_X))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=sample_X,
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=[
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
            ],
        )

        mv = register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            model_name=REGISTERED_MODEL_NAME,
            description=(
                f"Pipeline MiBici v2 | dataset={manifest_info['id']} | commit={commit_hash}"
            ),
            tags={
                "stage": "candidate",
                "dataset": "v2",
                "dataset_sha256": manifest_info["sha256"],
                "git_commit": commit_hash,
                "validation_passed": "true",
            },
        )
        logger.success(
            f"Modelo registrado en MLflow Registry como '{REGISTERED_MODEL_NAME}' "
            f"v{mv.version} (run={run.info.run_id})"
        )


if __name__ == "__main__":
    app()
