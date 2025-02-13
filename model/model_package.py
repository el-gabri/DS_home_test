import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import lightgbm as lgb
from pyod.models.copod import COPOD


class FraudDetectionModel:
    def __init__(self):
        self.supervised_model = None
        self.unsupervised_model = None
        self.scaler = None

    def preprocess_features(self, df):
        """Preprocess features for model input"""
        # Handle specific columns
        df['amount'] = df['amount'] / 100

        # Fill specific nulls with 0
        zero_fill_columns = [
            'mc_billpay_br_sum_last_15d',
            'mc_pur_br_sum_last_15d',
            'mc_billpay_br_sum_last_1y',
            'mc_pur_br_sum_last_1y',
            'mc_merchants_ip_count_dist_shared_last_15d'
        ]
        df[zero_fill_columns] = df[zero_fill_columns].fillna(0.0)

        return df

    def fit(self, X_train, y_train):
        """Train both supervised and unsupervised models"""
        # Initialize and fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # Train supervised model (LightGBM)
        self.supervised_model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=32,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            bagging_freq=5,
            verbose=-1
        )
        self.supervised_model.fit(X_scaled, y_train)

        # Train unsupervised model (COPOD)
        self.unsupervised_model = COPOD(contamination=0.05)
        self.unsupervised_model.fit(X_scaled)

    def predict(self, X):
        """Get predictions from both models"""
        X_scaled = self.scaler.transform(X)

        # Get supervised predictions
        supervised_proba = self.supervised_model.predict_proba(X_scaled)[:, 1]

        # Get unsupervised scores
        unsupervised_scores = self.unsupervised_model.decision_function(X_scaled)

        # Normalize unsupervised scores
        unsupervised_scores = (unsupervised_scores - np.min(unsupervised_scores)) / \
                              (np.max(unsupervised_scores) - np.min(unsupervised_scores))

        # Combine scores (70% supervised, 30% unsupervised)
        combined_scores = 0.7 * supervised_proba + 0.3 * unsupervised_scores

        return {
            'combined_score': combined_scores,
            'supervised_score': supervised_proba,
            'unsupervised_score': unsupervised_scores
        }

    def save_model(self, path):
        """Save model artifacts"""
        model_artifacts = {
            'supervised_model': self.supervised_model,
            'unsupervised_model': self.unsupervised_model,
            'scaler': self.scaler
        }
        joblib.dump(model_artifacts, path)

    @classmethod
    def load_model(cls, path):
        """Load saved model"""
        model = cls()
        artifacts = joblib.load(path)
        model.supervised_model = artifacts['supervised_model']
        model.unsupervised_model = artifacts['unsupervised_model']
        model.scaler = artifacts['scaler']
        return model