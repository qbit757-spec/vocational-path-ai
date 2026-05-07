import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from datetime import datetime
import json
import shutil

class MLService:
    def __init__(self):
        self.model_dir = 'app/ml/assets'
        self.datasets_dir = 'app/ml/datasets'
        self.model_path = os.path.join(self.model_dir, 'decision_tree_model.joblib')
        self.features_path = os.path.join(self.model_dir, 'features.joblib')
        self.stats_path = os.path.join(self.model_dir, 'model_stats.json')
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)

    def save_dataset(self, filename: str, content: bytes):
        path = os.path.join(self.datasets_dir, filename)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def list_datasets(self):
        files = os.listdir(self.datasets_dir)
        return [{"filename": f, "size": os.path.getsize(os.path.join(self.datasets_dir, f))} for f in files if f.endswith('.csv')]

    def clean_data(self, df: pd.DataFrame):
        """
        Example cleaning process:
        - Handle missing values
        - Ensure columns R, I, A, S, E, C exist and are numeric
        - Normalize or scale if necessary
        """
        logs = []
        logs.append(f"Initial shape: {df.shape}")
        
        # Keep only necessary columns if they exist
        required_cols = ['R', 'I', 'A', 'S', 'E', 'C', 'Career_Category']
        existing_cols = [c for c in required_cols if c in df.columns]
        df = df[existing_cols]
        logs.append(f"Columns kept: {existing_cols}")

        # Drop rows with missing RIASEC scores
        ria_cols = ['R', 'I', 'A', 'S', 'E', 'C']
        before_drop = len(df)
        df = df.dropna(subset=[c for c in ria_cols if c in df.columns])
        logs.append(f"Dropped {before_drop - len(df)} rows with null RIASEC scores")

        # Fill missing Career_Category if it's the target
        if 'Career_Category' in df.columns:
            df = df.dropna(subset=['Career_Category'])
            logs.append(f"Rows after dropping null targets: {len(df)}")
        
        return df, logs

    async def train_from_files(self, filenames: list[str] = None):
        all_logs = []
        all_dfs = []

        if not filenames:
            # Use synthetic data if no files provided
            all_logs.append("No files provided. Generating synthetic dataset...")
            df = self.generate_synthetic_df(2000)
            all_dfs.append(df)
        else:
            for fname in filenames:
                path = os.path.join(self.datasets_dir, fname)
                if os.path.exists(path):
                    all_logs.append(f"Processing {fname}...")
                    df = pd.read_csv(path)
                    df_cleaned, logs = self.clean_data(df)
                    all_dfs.extend(logs)
                    all_dfs.append(df_cleaned)
                else:
                    all_logs.append(f"File {fname} not found, skipping.")

        if not all_dfs:
            raise Exception("No data available for training.")

        full_df = pd.concat(all_dfs, ignore_index=True)
        all_logs.append(f"Final training dataset size: {len(full_df)}")

        # Check if we have the target column
        if 'Career_Category' not in full_df.columns:
            all_logs.append("Target column 'Career_Category' missing. Using synthetic labeling logic...")
            full_df['Career_Category'] = full_df.apply(self._synthetic_label, axis=1)

        X = full_df[['R', 'I', 'A', 'S', 'E', 'C']]
        y = full_df['Career_Category']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = DecisionTreeClassifier(max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        joblib.dump(model, self.model_path)
        joblib.dump(['R', 'I', 'A', 'S', 'E', 'C'], self.features_path)
        
        stats = {
            "accuracy": float(accuracy),
            "n_samples": len(full_df),
            "trained_at": datetime.now().isoformat(),
            "classes": list(model.classes_),
            "logs": all_logs
        }
        
        with open(self.stats_path, 'w') as f:
            json.dump(stats, f)
            
        return stats

    def generate_synthetic_df(self, n_samples=2000):
        np.random.seed(42)
        data = np.random.randint(0, 11, size=(n_samples, 6))
        df = pd.DataFrame(data, columns=['R', 'I', 'A', 'S', 'E', 'C'])
        df['Career_Category'] = df.apply(self._synthetic_label, axis=1)
        return df

    def _synthetic_label(self, row):
        scores = {
            'Ingeniería / Tecnología': row['R'] + row['I'],
            'Ciencias de la Salud': row['I'] + row['S'],
            'Artes y Diseño': row['A'],
            'Ciencias Sociales / Educación': row['S'],
            'Negocios / Derecho': row['E'],
            'Administración / Contabilidad': row['C']
        }
        return max(scores, key=scores.get)

    def get_model_stats(self):
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'r') as f:
                return json.load(f)
        return None

    def explain_prediction(self, scores_dict: dict):
        """
        Traces the decision path for a specific input and returns an interpretable flow.
        """
        if not os.path.exists(self.model_path):
            return None
            
        model = joblib.load(self.model_path)
        features = joblib.load(self.features_path)
        
        # Prepare input
        X = np.array([[scores_dict.get(f, 0) for f in features]])
        
        # Get decision path
        node_indicator = model.decision_path(X)
        leaf_id = model.apply(X)[0]
        
        tree = model.tree_
        feature_names = features
        
        path = []
        node_indices = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
        
        for node_id in node_indices:
            if leaf_id == node_id:
                path.append({
                    "node_id": int(node_id),
                    "type": "leaf",
                    "prediction": str(model.classes_[np.argmax(tree.value[node_id])]),
                    "probability": float(np.max(tree.value[node_id]) / np.sum(tree.value[node_id]))
                })
            else:
                feature = feature_names[tree.feature[node_id]]
                threshold = float(tree.threshold[node_id])
                value = float(X[0, tree.feature[node_id]])
                
                path.append({
                    "node_id": int(node_id),
                    "type": "decision",
                    "feature": feature,
                    "threshold": round(threshold, 2),
                    "value": value,
                    "condition": f"{feature} {'<=' if value <= threshold else '>'} {round(threshold, 2)}"
                })
        
        return path

ml_service = MLService()
