import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

class MarketAnalysis:
    def __init__(self):
        self.model = LinearRegression()
        self.data = None
        
    def process_data(self, data: pd.DataFrame) -> None:
        """Process market data for analysis."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame.")
        self.data = data.dropna().copy()
        
    def predict_trend(self) -> pd.Series:
        """Predict market trends using linear regression."""
        try:
            if self.data.empty:
                return pd.Series()
            y = self.data['price']
            X = self.data[['volume', 'demand']]
            self.model.fit(X, y)
            predictions = self.model.predict(X)
            return pd.Series(predictions, index=self.data.index, name='predicted_price')
        except Exception as e:
            print(f"Error in prediction: {e}")
            raise

    def evaluate_model(self) -> float:
        """Evaluate model performance."""
        if not hasattr(self, 'model'):
            raise AttributeError("Model has not been trained.")
        y_true = self.data['price']
        X = self.data[['volume', 'demand']]
        try:
            predictions = self.model.predict(X)
            return mean_absolute_error(y_true, predictions)
        except Exception as e:
            print(f"Error in evaluation: {e}")
            raise