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
from typing import List, Dict, Any, Optional

class MLService:
    def __init__(self):
        # Use absolute paths to avoid issues in Docker
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # points to app/
        self.ml_dir = os.path.join(self.base_dir, 'ml')
        self.model_dir = os.path.join(self.ml_dir, 'assets')
        self.datasets_dir = os.path.join(self.ml_dir, 'datasets')
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        
        self.model_path = os.path.join(self.model_dir, 'decision_tree_model.joblib')
        self.features_path = os.path.join(self.model_dir, 'features.joblib')
        self.stats_path = os.path.join(self.model_dir, 'model_stats.json')

    def save_dataset(self, filename: str, content: bytes):
        path = os.path.join(self.datasets_dir, filename)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def list_datasets(self):
        if not os.path.exists(self.datasets_dir):
            return []
        files = os.listdir(self.datasets_dir)
        return [{"filename": f, "size": os.path.getsize(os.path.join(self.datasets_dir, f))} for f in files if f.endswith('.csv')]

    def clean_data(self, df: pd.DataFrame):
        """
        Processes real-world data from OpenPsychometrics or synthetic data.
        """
        logs = []
        logs.append(f"Initial shape: {df.shape}")
        
        # Check if it's the 48-item RIASEC format
        is_real_riasec = all(f"{cat}{i}" in df.columns for cat in 'RISEC' for i in range(1, 9)) and 'A1' in df.columns
        
        if is_real_riasec:
            logs.append("Detected real-world 48-item RIASEC format. Starting specialized cleaning...")
            
            # 1. VCL Validity Check (Honest check)
            # VCL6, VCL9, VCL12 are non-existent words. If checked (1), the entry is invalid.
            if all(c in df.columns for c in ['VCL6', 'VCL9', 'VCL12']):
                before_vcl = len(df)
                df = df[(df['VCL6'] == 0) & (df['VCL9'] == 0) & (df['VCL12'] == 0)]
                logs.append(f"Dropped {before_vcl - len(df)} invalid records (failed VCL honesty check)")
            
            # 2. Age Filter (> 13 as per dataset description)
            if 'age' in df.columns:
                before_age = len(df)
                # Keep ages between 13 and 80 to avoid extreme outliers
                df = df[(df['age'] >= 13) & (df['age'] <= 80)]
                logs.append(f"Dropped {before_age - len(df)} records with invalid age")

            # 3. Aggregate RIASEC items (Sum R1-R8, etc.)
            for cat in 'RIASEC':
                cols = [f"{cat}{i}" for i in range(1, 9)]
                # Aggregate and normalize to 0-10 scale
                df[cat] = df[cols].sum(axis=1)
                # Normalizing: (Sum - min_possible) / (max_possible - min_possible) * 10
                # Min possible sum is 8 (all 1s), Max is 40 (all 5s)
                df[cat] = ((df[cat] - 8) / (40 - 8)) * 10
            
            # 4. Map 'major' to 'Career_Category'
            if 'major' in df.columns:
                df['Career_Category'] = df['major'].apply(self._map_major_to_category)
                df = df.dropna(subset=['Career_Category'])
                logs.append(f"Mapped majors to categories. Final rows: {len(df)}")
        else:
            # Handle synthetic or pre-processed data (columns R, I, A, S, E, C)
            required_cols = ['R', 'I', 'A', 'S', 'E', 'C', 'Career_Category']
            existing_cols = [c for c in required_cols if c in df.columns]
            df = df[existing_cols]
            
            ria_cols = ['R', 'I', 'A', 'S', 'E', 'C']
            df = df.dropna(subset=[c for c in ria_cols if c in df.columns])
        
        return df, logs

    def _map_major_to_category(self, major: Any) -> Optional[str]:
        if not isinstance(major, str): return None
        major = major.lower()
        
        mapping = {
            'Ingeniería / Tecnología': ['eng', 'comp', 'tech', 'software', 'civil', 'mech', 'it', 'math', 'physic'],
            'Ciencias de la Salud': ['med', 'nurs', 'dent', 'bio', 'pharm', 'psych', 'health', 'vet'],
            'Artes y Diseño': ['art', 'design', 'music', 'dance', 'fashion', 'film', 'photo', 'paint'],
            'Ciencias Sociales / Educación': ['edu', 'teach', 'soc', 'hist', 'law', 'polit', 'anthro', 'ling'],
            'Negocios / Derecho': ['bus', 'market', 'econ', 'law', 'finan', 'entre', 'trade'],
            'Administración / Contabilidad': ['admin', 'acc', 'audit', 'manage', 'office', 'hr']
        }
        
        for category, keywords in mapping.items():
            if any(k in major for k in keywords):
                return category
        return None

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
                    all_logs.extend(logs)
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
        Also includes the full tree structure for visualization and advanced insights.
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
        
        # Get the full tree map
        full_tree = self.get_full_tree_structure(model)
        
        # Calculate probabilities for all classes at that leaf
        class_probs = tree.value[leaf_id][0] / np.sum(tree.value[leaf_id][0])
        sorted_indices = np.argsort(class_probs)[::-1]
        
        main_prediction = str(model.classes_[sorted_indices[0]])
        main_confidence = float(class_probs[sorted_indices[0]])
        
        second_prediction = str(model.classes_[sorted_indices[1]]) if len(sorted_indices) > 1 else None
        second_confidence = float(class_probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0
        
        # Conflict detection: If the difference is small
        has_conflict = (main_confidence - second_confidence) < 0.20
        
        return {
            "decision_path": path,
            "full_tree": full_tree,
            "leaf_id": int(leaf_id),
            "insights": {
                "confidence": round(main_confidence * 100, 2),
                "is_multipotential": has_conflict,
                "second_option": {
                    "career": second_prediction,
                    "confidence": round(second_confidence * 100, 2)
                },
                "analysis": "Perfil con alta claridad vocacional" if not has_conflict else "Perfil multipotencial: Se recomienda entrevista profunda para decidir entre las dos primeras opciones."
            }
        }

    def get_full_tree_structure(self, model=None) -> Dict[str, Any]:
        if model is None:
            if not os.path.exists(self.model_path):
                return {}
            model = joblib.load(self.model_path)
            
        tree = model.tree_
        features = joblib.load(self.features_path)
        
        def recurse(node: int) -> Dict[str, Any]:
            if tree.children_left[node] == tree.children_right[node]:  # Leaf
                return {
                    "node_id": int(node),
                    "type": "leaf",
                    "prediction": str(model.classes_[np.argmax(tree.value[node])])
                }
            else:  # Decision node
                return {
                    "node_id": int(node),
                    "type": "decision",
                    "feature": features[tree.feature[node]],
                    "threshold": round(float(tree.threshold[node]), 2),
                    "left": recurse(tree.children_left[node]),
                    "right": recurse(tree.children_right[node])
                }
                
        return recurse(0)

ml_service = MLService()
