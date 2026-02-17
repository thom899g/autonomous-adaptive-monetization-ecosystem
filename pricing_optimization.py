import numpy as np
from xgboost import XGBRegressor

class PricingOptimizer:
    def __init__(self):
        self.model = XGBRegressor()
        
    def train_model(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the pricing model."""
        if not isinstance(features, np.ndarray) or not isinstance(labels, np.ndarray):
            raise TypeError("Features and labels must be numpy arrays.")
        self.model.fit(features, labels)
        
    def predict_price(self, input_features: np.ndarray) -> np.ndarray:
        """Predict optimal prices based on features."""
        try:
            if input_features.shape[1] != self.model.n_features_in_:
                raise ValueError("Input features do not match model's feature count.")
            return self.model.predict(input_features)
        except Exception as e:
            print(f"Pricing error: {e}")
            raise
            
    def validate_model(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Validate the model against a test set."""
        try:
            if not hasattr(self, 'model'):
                raise AttributeError("Model has not been trained.")
            return self.model.score(features, labels)
        except Exception as e:
            print(f"Validation error: {e}")
            raise