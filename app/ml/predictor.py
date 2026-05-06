import joblib
import os
import numpy as np

class VocationalPredictor:
    def __init__(self):
        self.model_path = 'app/ml/assets/decision_tree_model.joblib'
        self.features_path = 'app/ml/assets/features.joblib'
        self.model = None
        self.features = None
        
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.features = joblib.load(self.features_path)
            return True
        return False

    def predict(self, scores):
        """
        scores: dict with RIASEC keys (R, I, A, S, E, C)
        """
        if self.model is None:
            if not self.load_model():
                raise Exception("Model not found. Please train the model first.")
        
        # Prepare input vector
        input_vector = [scores.get(f, 0) for f in self.features]
        prediction = self.model.predict([input_vector])
        return prediction[0]

predictor = VocationalPredictor()
