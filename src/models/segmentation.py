"""
Customer Segmentation Module using K-Means Clustering and PCA.
Evaluates silhouette scores, assigns data-driven business labels based on actual centroid profiles,
and generates recommended business retention actions.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib
import logging

from src.config import RANDOM_SEED, MODEL_ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def derive_business_label(centroid: Dict[str, float], overall_means: Dict[str, float]) -> Tuple[str, str, str]:
    """
    Assigns human-interpretable business labels based on actual measured cluster characteristics.
    Returns (label, description, recommended_action).
    """
    tenure = centroid.get("tenure", 0)
    clv = centroid.get("discounted_clv", 0)
    churn_prob = centroid.get("churn_probability", 0)
    monthly = centroid.get("monthly_charges", 0)

    mean_clv = overall_means.get("discounted_clv", 500)
    mean_tenure = overall_means.get("tenure", 32)
    mean_churn = overall_means.get("churn_probability", 0.27)

    if clv >= mean_clv and churn_prob >= mean_churn:
        return (
            "High-Value At-Risk",
            "High lifetime contribution margin with elevated probability of churn. Critical revenue threat.",
            "Immediate proactive outreach, tailored renewal incentive, and VIP customer service escalation.",
        )
    elif clv >= mean_clv and churn_prob < mean_churn:
        return (
            "High-Value Loyal Champions",
            "Long-tenure, high-spending accounts with low churn risk. Core revenue bedrock.",
            "Loyalty rewards, exclusive early access to new features, and referral incentive programs.",
        )
    elif tenure < mean_tenure and churn_prob >= mean_churn:
        return (
            "Early-Tenure High-Risk",
            "Recent sign-ups experiencing onboarding friction with low commitment.",
            "Onboarding check-ins, feature adoption tutorials, and contract upgrade incentives.",
        )
    else:
        return (
            "Budget Steady Savers",
            "Price-sensitive customers with moderate tenure and stable baseline retention.",
            "Targeted cross-sell campaigns for high-value add-ons (streaming bundles, security).",
        )


class CustomerSegmentationService:
    """Performs K-Means clustering, silhouette optimization, and 2D PCA projection."""

    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.scaler: Optional[StandardScaler] = None
        self.kmeans: Optional[KMeans] = None
        self.pca: Optional[PCA] = None
        self.cluster_metadata: Dict[int, Dict[str, Any]] = {}
        self.silhouette_scores: Dict[int, float] = {}
        self.feature_cols: List[str] = [
            "tenure",
            "monthly_charges",
            "total_charges",
            "num_services_subscribed",
            "churn_probability",
            "discounted_clv",
        ]

    def fit(self, df: pd.DataFrame) -> "CustomerSegmentationService":
        """
        Fits StandardScaler, selects optimal K based on silhouette analysis, and fits final K-Means.
        """
        logger.info("Initializing Customer Segmentation clustering...")
        
        # Verify required clustering features exist
        cluster_data = df[self.feature_cols].copy().fillna(0)

        # 1. Scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(cluster_data)

        # 2. Evaluate K from 3 to 6
        best_k = self.n_clusters
        best_score = -1.0
        for k in [3, 4, 5, 6]:
            km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
            labels = km.fit_predict(X_scaled)
            score = float(silhouette_score(X_scaled, labels))
            self.silhouette_scores[k] = round(score, 4)
            if score > best_score:
                best_score = score
                best_k = k

        logger.info(f"Silhouette scores: {self.silhouette_scores}. Selected optimal k={best_k}")
        self.n_clusters = best_k

        # 3. Fit optimal KMeans
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=RANDOM_SEED, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_scaled)

        # 4. Fit 2D PCA for dashboard visualization
        self.pca = PCA(n_components=2, random_state=RANDOM_SEED)
        self.pca.fit(X_scaled)

        # 5. Compute centroids and derive descriptive business labels
        cluster_data["cluster_id"] = cluster_labels
        overall_means = cluster_data[self.feature_cols].mean().to_dict()

        for c_id in range(self.n_clusters):
            c_mask = cluster_data["cluster_id"] == c_id
            c_df = cluster_data[c_mask]
            centroid = c_df[self.feature_cols].mean().to_dict()

            label, desc, action = derive_business_label(centroid, overall_means)
            self.cluster_metadata[c_id] = {
                "cluster_id": c_id,
                "cluster_label": label,
                "cluster_description": desc,
                "recommended_action": action,
                "customer_count": int(c_mask.sum()),
                "customer_pct": round(float(c_mask.mean() * 100), 1),
                "centroid": {k: round(float(v), 2) for k, v in centroid.items()},
            }

        logger.info(f"Customer segmentation fitted successfully with {self.n_clusters} clusters.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns cluster labels, descriptions, and 2D PCA coordinates to input records.
        """
        if self.kmeans is None or self.scaler is None or self.pca is None:
            raise ValueError("Segmentation service must be fitted before transform.")

        df_proc = df.copy()
        if "num_services_subscribed" not in df_proc.columns:
            from src.data.features import engineer_features
            df_proc = engineer_features(df_proc)
            if "num_services_subscribed" not in df_proc.columns:
                df_proc["num_services_subscribed"] = 0

        for col in self.feature_cols:
            if col not in df_proc.columns:
                df_proc[col] = 0.0

        cluster_data = df_proc[self.feature_cols].copy().fillna(0)
        X_scaled = self.scaler.transform(cluster_data)
        cluster_ids = self.kmeans.predict(X_scaled)
        pca_coords = self.pca.transform(X_scaled)

        out = df_proc.copy()
        out["cluster_id"] = cluster_ids
        out["cluster_label"] = [self.cluster_metadata[c]["cluster_label"] for c in cluster_ids]
        out["cluster_description"] = [self.cluster_metadata[c]["cluster_description"] for c in cluster_ids]
        out["recommended_action"] = [self.cluster_metadata[c]["recommended_action"] for c in cluster_ids]
        out["pca_dim_1"] = np.round(pca_coords[:, 0], 3)
        out["pca_dim_2"] = np.round(pca_coords[:, 1], 3)

        return out

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Saves segmentation artifacts to disk."""
        save_path = filepath or (MODEL_ARTIFACTS_DIR / "segmentation_model.joblib")
        joblib.dump(self, save_path)
        logger.info(f"Saved customer segmentation models to {save_path}")
        return save_path

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "CustomerSegmentationService":
        """Loads fitted customer segmentation models."""
        load_path = filepath or (MODEL_ARTIFACTS_DIR / "segmentation_model.joblib")
        if not Path(load_path).exists():
            raise FileNotFoundError(f"Segmentation artifact not found at {load_path}")
        return joblib.load(load_path)
