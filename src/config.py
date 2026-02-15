# src/config.py
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    data_path: str = os.getenv("DATA_PATH", "data/creditcard.csv")
    reports_dir: str = os.getenv("REPORTS_DIR", "reports")
    models_dir: str = os.getenv("MODELS_DIR", "models")
    seed: int = int(os.getenv("SEED", "42"))

    target_col: str = os.getenv("TARGET_COL", "Class")
    time_col: str = os.getenv("TIME_COL", "Time")
    amount_col: str = os.getenv("AMOUNT_COL", "Amount")
