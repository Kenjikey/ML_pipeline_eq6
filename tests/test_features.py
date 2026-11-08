"""
Unit tests for Feature Engineering module (cyclical encoding and station lags).
"""

import unittest
import pandas as pd
import numpy as np

from mibici.features import add_calendar_and_cyclical_features, add_historical_lags_and_rolling

class TestFeatures(unittest.TestCase):
    def setUp(self):
        # Dataset horario sintético con estacion_id, entradas, salidas y balance
        dates = pd.date_range("2023-01-01 00:00:00", "2023-01-10 23:00:00", freq="1h")
        self.sample_df = pd.DataFrame({
            "estacion_id": [1] * len(dates),
            "timestamp": dates,
            "entradas": np.random.poisson(lam=3, size=len(dates)),
            "salidas": np.random.poisson(lam=3, size=len(dates)),
            "balance": np.random.randint(-3, 4, size=len(dates))
        })

    def test_calendar_and_cyclical_features(self):
        """Verifica que las variables ciclicas (sin/cos) y de calendario se generen correctamente."""
        df_feat = add_calendar_and_cyclical_features(self.sample_df)
        
        # Validar existencia de columnas ciclicas generadas
        columnas_esperadas = ["hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos", "es_fin_de_semana"]
        for col in columnas_esperadas:
            self.assertIn(col, df_feat.columns)
            
        # Validar que los valores sin/cos esten en el rango matematico [-1.0, 1.0]
        for col in ["hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos"]:
            self.assertTrue((df_feat[col] >= -1.0).all() and (df_feat[col] <= 1.0).all())

    def test_historical_lags_and_rolling(self):
        """Verifica que se generen las columnas de retardos temporales (lags) y medias moviles."""
        df_lags = add_historical_lags_and_rolling(self.sample_df)
        
        # Validar que existan columnas de lags y rolling
        lag_cols = [c for c in df_lags.columns if "lag" in c or "roll" in c or "media" in c or "std" in c]
        self.assertGreater(len(lag_cols), 0)

if __name__ == "__main__":
    unittest.main()