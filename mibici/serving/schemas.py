"""Contrato de entrada y salida de la API de prediccion MiBici."""

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class StationHourFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    estacion_id: int = Field(..., ge=1, examples=[51])
    mes: int = Field(..., ge=1, le=12, examples=[11])
    dia_mes: int = Field(..., ge=1, le=31, examples=[15])
    dia_semana: int = Field(..., ge=0, le=6, examples=[2])
    hora: int = Field(..., ge=0, le=23, examples=[8])
    es_fin_de_semana: int = Field(..., ge=0, le=1, examples=[0])
    es_hora_pico: int = Field(..., ge=0, le=1, examples=[1])
    hora_sin: float = Field(..., ge=-1.0, le=1.0, examples=[0.866025])
    hora_cos: float = Field(..., ge=-1.0, le=1.0, examples=[-0.5])
    dia_semana_sin: float = Field(..., ge=-1.0, le=1.0, examples=[0.974928])
    dia_semana_cos: float = Field(..., ge=-1.0, le=1.0, examples=[-0.222521])

    entradas_lag_1h: float = Field(..., ge=0.0, examples=[6.0])
    entradas_lag_2h: float = Field(..., ge=0.0, examples=[4.0])
    entradas_lag_24h: float = Field(..., ge=0.0, examples=[7.0])
    entradas_lag_168h: float = Field(..., ge=0.0, examples=[5.0])

    salidas_lag_1h: float = Field(..., ge=0.0, examples=[3.0])
    salidas_lag_2h: float = Field(..., ge=0.0, examples=[2.0])
    salidas_lag_24h: float = Field(..., ge=0.0, examples=[4.0])
    salidas_lag_168h: float = Field(..., ge=0.0, examples=[3.0])

    balance_lag_1h: float = Field(..., examples=[3.0])
    balance_lag_2h: float = Field(..., examples=[2.0])
    balance_lag_24h: float = Field(..., examples=[3.0])
    balance_lag_168h: float = Field(..., examples=[2.0])

    entradas_roll_mean_3h: float = Field(..., ge=0.0, examples=[4.33])
    entradas_roll_mean_6h: float = Field(..., ge=0.0, examples=[3.5])
    salidas_roll_mean_3h: float = Field(..., ge=0.0, examples=[2.33])
    salidas_roll_mean_6h: float = Field(..., ge=0.0, examples=[2.0])
    balance_roll_mean_3h: float = Field(..., examples=[2.0])
    balance_roll_mean_6h: float = Field(..., examples=[1.5])


class PredictionItem(BaseModel):
    estacion_id: int
    pred_entradas: float
    pred_salidas: float
    pred_balance: float
    accion_operativa: str


class BatchPredictionRequest(BaseModel):
    records: List[StationHourFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    model_version: str
    count: int
    predictions: List[PredictionItem]
