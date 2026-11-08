"""
Unit tests for data validation and basic structures.
"""

import unittest
import pandas as pd
import numpy as np

class TestDataIntegrity(unittest.TestCase):
    def test_sample_dataframe_structure(self):
        """Valida que los tipos de datos basicos de viajes tengan el formato requerido."""
        df = pd.DataFrame({
            "Viaje_Id": [100, 101],
            "Inicio_del_viaje": ["2023-01-01 08:00:00", "2023-01-01 08:30:00"],
            "Fin_del_viaje": ["2023-01-01 08:15:00", "2023-01-01 08:45:00"],
            "Origen_Id": [1, 2],
            "Destino_Id": [2, 1]
        })
        
        df["Inicio_del_viaje"] = pd.to_datetime(df["Inicio_del_viaje"])
        df["Fin_del_viaje"] = pd.to_datetime(df["Fin_del_viaje"])
        duraciones = (df["Fin_del_viaje"] - df["Inicio_del_viaje"]).dt.total_seconds()
        
        self.assertEqual(len(df), 2)
        self.assertTrue((duraciones > 0).all())

if __name__ == "__main__":
    unittest.main()