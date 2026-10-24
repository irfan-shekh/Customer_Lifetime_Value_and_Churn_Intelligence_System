"""
Data preprocessing module.
Handles stratified train/val/test splits and builds leak-free preprocessing pipelines.
Never fits transformations on held-out validation or test splits.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import logging

from src.data.features import engineer_features, get_feature_metadata
from src.config import RANDOM_SEED, TEST_SIZE, VAL_SIZE, MODEL_ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Leak-free feature preprocessing transformer.
    Fits OneHotEncoder and StandardScaler on training records ONLY.
    """

    def __init__(self):
        self.numeric_features, self.categorical_features, self.dropped_cols = get_feature_metadata()
        self.transformer: Optional[ColumnTransformer] = None
        self.feature_names_out: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame) -> "DataPreprocessor":
        """Fits encoder and scaler on training features only."""
        # Ensure engineered features exist
        X_eng = engineer_features(X)
        
        # Verify available features
        num_cols = [c for c in self.numeric_features if c in X_eng.columns]
        cat_cols = [c for c in self.categorical_features if c in X_eng.columns]

        self.transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ],
            remainder="drop",
        )

        self.transformer.fit(X_eng)
        
        # Extract transformed column names
        encoded_cat_names = self.transformer.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
        self.feature_names_out = num_cols + encoded_cat_names
        self.is_fitted = True
        logger.info(f"Preprocessor fitted with {len(self.feature_names_out)} final transformed features.")
        return self

    def transform(self, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Transforms input features using fitted encoders/scalers."""
        if not self.is_fitted or self.transformer is None:
            raise ValueError("DataPreprocessor must be fitted before calling transform().")
        
        X_eng = engineer_features(X)
        X_transformed = self.transformer.transform(X_eng)
        return X_transformed, self.feature_names_out

    def fit_transform(self, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        return self.fit(X).transform(X)

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Saves fitted preprocessor artifact to disk."""
        save_path = filepath or (MODEL_ARTIFACTS_DIR / "preprocessor.joblib")
        joblib.dump(self, save_path)
        logger.info(f"Saved preprocessor to {save_path}")
        return save_path

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "DataPreprocessor":
        """Loads fitted preprocessor from disk."""
        load_path = filepath or (MODEL_ARTIFACTS_DIR / "preprocessor.joblib")
        if not Path(load_path).exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {load_path}")
        return joblib.load(load_path)


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified train, validation, and test split based on Churn status.
    Ensures held-out splits are completely untouched during preprocessing and fitting.
    """
    if "churn" not in df.columns:
        raise ValueError("DataFrame must contain 'churn' column for stratified splitting.")

    # First split: Train+Val vs Test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["churn"],
    )

    # Second split: Train vs Val
    # val_size represents fraction of original train_val set
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val_df["churn"],
    )

    logger.info(
        f"Data split completed: Train={len(train_df)} ({len(train_df)/len(df):.1%}), "
        f"Val={len(val_df)} ({len(val_df)/len(df):.1%}), "
        f"Test={len(test_df)} ({len(test_df)/len(df):.1%})"
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
