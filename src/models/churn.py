"""
Churn Prediction Modeling Module.
Trains LightGBM classifier with class imbalance handling, baseline Logistic Regression comparison,
business-driven threshold tuning on validation set, and persistent artifact saving.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)
import joblib
import logging

from src.data.preprocess import DataPreprocessor, split_data
from src.config import RANDOM_SEED, MODEL_ARTIFACTS_DIR, DEFAULT_CHURN_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_optimal_threshold(
    y_val: np.ndarray,
    y_val_probs: np.ndarray,
    cost_per_offer: float = 20.0,
    value_per_save: float = 120.0,
    uplift_retention_rate: float = 0.35,
) -> Tuple[float, Dict[str, Any]]:
    """
    Finds decision threshold optimizing expected net business value on validation split.
    Trade-off:
      - False Positive (Targeting stayer): Incurs cost_per_offer without saving anyone.
      - True Positive (Targeting churner): Incurs cost_per_offer and saves customer with uplift_retention_rate.
      - False Negative (Missing churner): Loses value_per_save.
    """
    thresholds = np.linspace(0.10, 0.85, 76)
    best_thresh = 0.50
    max_net_value = -float("inf")
    analysis_records = []

    total_actual_churners = int(np.sum(y_val))

    for t in thresholds:
        y_pred = (y_val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
        
        # Financial simulation on validation cohort
        total_targeted = tp + fp
        total_campaign_cost = total_targeted * cost_per_offer
        expected_saved = tp * uplift_retention_rate
        expected_revenue_saved = expected_saved * value_per_save
        net_value = expected_revenue_saved - total_campaign_cost

        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        analysis_records.append({
            "threshold": round(float(t), 3),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "net_value": round(float(net_value), 2),
            "targeted_count": int(total_targeted),
            "expected_saved": round(float(expected_saved), 1),
        })

        if net_value > max_net_value:
            max_net_value = net_value
            best_thresh = t

    return round(float(best_thresh), 3), {
        "best_threshold": round(float(best_thresh), 3),
        "max_net_value": round(float(max_net_value), 2),
        "threshold_analysis": analysis_records,
    }


def evaluate_model(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Computes comprehensive binary classification metrics."""
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_probs)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_probs)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_probs)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }


class ChurnModelService:
    """Encapsulates training, validation, threshold selection, and prediction for Churn."""

    def __init__(self):
        self.preprocessor: Optional[DataPreprocessor] = None
        self.lgb_model: Optional[lgb.LGBMClassifier] = None
        self.baseline_model: Optional[LogisticRegression] = None
        self.best_threshold: float = DEFAULT_CHURN_THRESHOLD
        self.metrics: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.threshold_report: Dict[str, Any] = {}

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end reproducible training workflow:
        1. Fits preprocessor on train_df ONLY.
        2. Fits Logistic Regression baseline.
        3. Fits LightGBM with early stopping on val set.
        4. Optimizes decision threshold using validation business objective.
        5. Computes held-out test evaluation metrics for both models.
        """
        logger.info("Initializing Churn Model Training Workflow...")

        # 1. Feature Preprocessing
        self.preprocessor = DataPreprocessor()
        X_train, self.feature_names = self.preprocessor.fit_transform(train_df)
        y_train = train_df["churn"].values

        X_val, _ = self.preprocessor.transform(val_df)
        y_val = val_df["churn"].values

        X_test, _ = self.preprocessor.transform(test_df)
        y_test = test_df["churn"].values

        # Class imbalance calculation
        neg_count = int(np.sum(y_train == 0))
        pos_count = int(np.sum(y_train == 1))
        scale_pos_weight = round(neg_count / max(pos_count, 1), 2)
        logger.info(f"Class imbalance: Negatives={neg_count}, Positives={pos_count}, ScalePosWeight={scale_pos_weight}")

        # 2. Baseline Logistic Regression
        logger.info("Training Logistic Regression Baseline...")
        self.baseline_model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
        )
        self.baseline_model.fit(X_train, y_train)
        baseline_test_probs = self.baseline_model.predict_proba(X_test)[:, 1]
        baseline_metrics = evaluate_model(y_test, baseline_test_probs, threshold=0.50)

        # 3. LightGBM Classifier
        logger.info("Training Primary LightGBM Classifier...")
        self.lgb_model = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.5,  # Moderate balance adjustment
            random_state=RANDOM_SEED,
            verbosity=-1,
        )

        self.lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        # 4. Threshold tuning on Validation Set
        val_probs = self.lgb_model.predict_proba(X_val)[:, 1]
        self.best_threshold, self.threshold_report = find_optimal_threshold(
            y_val,
            val_probs,
        )
        logger.info(f"Optimal business threshold determined on validation set: {self.best_threshold}")

        # 5. Held-out Test Set Evaluation
        test_probs = self.lgb_model.predict_proba(X_test)[:, 1]
        
        # Evaluate LightGBM at standard 0.50 and tuned threshold
        lgb_standard_metrics = evaluate_model(y_test, test_probs, threshold=0.50)
        lgb_tuned_metrics = evaluate_model(y_test, test_probs, threshold=self.best_threshold)

        self.metrics = {
            "baseline_logistic_regression": baseline_metrics,
            "lightgbm_standard_threshold_0_50": lgb_standard_metrics,
            "lightgbm_tuned_threshold": lgb_tuned_metrics,
            "scale_pos_weight": scale_pos_weight,
            "test_sample_size": len(y_test),
            "test_churn_rate": round(float(np.mean(y_test)), 4),
        }

        logger.info(
            f"Test ROC-AUC: LightGBM={lgb_tuned_metrics['roc_auc']} vs Baseline={baseline_metrics['roc_auc']}. "
            f"Tuned Recall={lgb_tuned_metrics['recall']} (F1={lgb_tuned_metrics['f1_score']})"
        )

        return self.metrics

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Generates churn probability estimates for customer records."""
        if self.lgb_model is None or self.preprocessor is None:
            raise ValueError("ChurnModelService must be trained or loaded before prediction.")
        X, _ = self.preprocessor.transform(df)
        return self.lgb_model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Generates binary risk flag (1=High Risk, 0=Low Risk) at specified or tuned threshold."""
        thresh = threshold if threshold is not None else self.best_threshold
        probs = self.predict_proba(df)
        return (probs >= thresh).astype(int)

    def save(self, directory: Optional[Path] = None) -> Path:
        """Saves model artifacts, preprocessor, and metadata to disk."""
        save_dir = directory or MODEL_ARTIFACTS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model object
        model_path = save_dir / "churn_lgb_model.joblib"
        joblib.dump(self.lgb_model, model_path)

        # Save baseline
        baseline_path = save_dir / "churn_baseline_lr.joblib"
        joblib.dump(self.baseline_model, baseline_path)

        # Save preprocessor
        self.preprocessor.save(save_dir / "preprocessor.joblib")

        # Save metadata and metrics
        metadata = {
            "best_threshold": self.best_threshold,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "threshold_report": self.threshold_report,
        }
        metadata_path = save_dir / "churn_model_metadata.joblib"
        joblib.dump(metadata, metadata_path)

        logger.info(f"All churn model artifacts successfully saved to {save_dir}")
        return save_dir

    @classmethod
    def load(cls, directory: Optional[Path] = None) -> "ChurnModelService":
        """Loads fitted churn model service and associated preprocessors."""
        load_dir = directory or MODEL_ARTIFACTS_DIR
        service = cls()
        
        model_path = load_dir / "churn_lgb_model.joblib"
        meta_path = load_dir / "churn_model_metadata.joblib"
        prep_path = load_dir / "preprocessor.joblib"

        if not model_path.exists() or not meta_path.exists() or not prep_path.exists():
            raise FileNotFoundError(f"Missing required churn artifacts in {load_dir}")

        service.lgb_model = joblib.load(model_path)
        service.preprocessor = DataPreprocessor.load(prep_path)
        
        meta = joblib.load(meta_path)
        service.best_threshold = meta.get("best_threshold", DEFAULT_CHURN_THRESHOLD)
        service.metrics = meta.get("metrics", {})
        service.feature_names = meta.get("feature_names", [])
        service.threshold_report = meta.get("threshold_report", {})

        baseline_path = load_dir / "churn_baseline_lr.joblib"
        if baseline_path.exists():
            service.baseline_model = joblib.load(baseline_path)

        logger.info(f"Loaded churn model from {load_dir} with threshold={service.best_threshold}")
        return service
