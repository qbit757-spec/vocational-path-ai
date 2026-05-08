import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from datetime import datetime
import json
import shutil
import traceback
from typing import List, Dict, Any, Optional

class MLService:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.ml_dir = os.path.join(self.base_dir, 'ml')
        self.model_dir = os.path.join(self.ml_dir, 'assets')
        self.datasets_dir = os.path.join(self.ml_dir, 'datasets')
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        
        # We store two models: one for Accuracy (RF) and one for XAI (Tree)
        self.model_path = os.path.join(self.model_dir, 'decision_tree_model.joblib')
        self.rf_model_path = os.path.join(self.model_dir, 'random_forest_model.joblib')
        self.features_path = os.path.join(self.model_dir, 'features.joblib')
        self.stats_path = os.path.join(self.model_dir, 'model_stats.json')

    def list_datasets(self):
        if not os.path.exists(self.datasets_dir): return []
        files = os.listdir(self.datasets_dir)
        return [{"filename": f, "size": os.path.getsize(os.path.join(self.datasets_dir, f))} for f in files if f.endswith('.csv')]

    def clean_data(self, df: pd.DataFrame):
        logs = []
        df.columns = [str(c).strip().replace('"', '') for c in df.columns]
        
        ria_map = {
            'R': [c for c in df.columns if c.startswith('R') and c[1:].isdigit()],
            'I': [c for c in df.columns if c.startswith('I') and c[1:].isdigit()],
            'A': [c for c in df.columns if c.startswith('A') and c[1:].isdigit()],
            'S': [c for c in df.columns if c.startswith('S') and c[1:].isdigit()],
            'E': [c for c in df.columns if (c.startswith('E') or c.startswith('F')) and c[1:].isdigit()],
            'C': [c for c in df.columns if c.startswith('C') and c[1:].isdigit()]
        }
        
        if all(len(cols) >= 5 for cols in ria_map.values()):
            # Cleaning logic
            if 'VCL6' in df.columns:
                df = df[pd.to_numeric(df['VCL6'], errors='coerce').fillna(0) == 0]
            if 'age' in df.columns:
                df['age'] = pd.to_numeric(df['age'], errors='coerce')
                df = df[(df['age'] >= 14) & (df['age'] <= 75)]
            
            for cat, cols in ria_map.items():
                for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(3)
                df[cat] = ((df[cols].mean(axis=1) - 1) / 4) * 10
            
            if 'major' in df.columns:
                df['Career_Category'] = df['major'].apply(self._map_major_to_category)
                df = df.dropna(subset=['Career_Category'])
                logs.append(f"Cleaned data: {len(df)} samples.")
        
        return df, logs

    def _map_major_to_category(self, major: Any) -> Optional[str]:
        if not isinstance(major, str): return None
        m = str(major).lower().strip()
        if not m or m == 'nan': return None
        
        mapping = {
            'Ingeniería / Tecnología': ['eng', 'comp', 'tech', 'soft', 'civil', 'mech', 'it', 'math', 'phys', 'syst', 'scie', 'data', 'web', 'electr', 'robot', 'arch', 'mining', 'telecom', 'indust'],
            'Ciencias de la Salud': ['med', 'nurs', 'dent', 'bio', 'phar', 'psyc', 'heal', 'vet', 'thera', 'medic', 'nurse', 'doct', 'physio', 'biol', 'nutri', 'kine', 'obs'],
            'Artes y Diseño': ['art', 'desig', 'musi', 'danc', 'fash', 'film', 'phot', 'pain', 'lite', 'crea', 'writ', 'dram', 'thea', 'fine', 'graph', 'visu', 'animat'],
            'Ciencias Sociales / Educación': ['edu', 'teac', 'soc', 'hist', 'law', 'poli', 'anth', 'ling', 'phil', 'coun', 'comm', 'phil', 'geog', 'inter', 'journa', 'sociol', 'human'],
            'Negocios / Derecho': ['bus', 'mark', 'econ', 'law', 'fina', 'entr', 'trad', 'comm', 'busi', 'lega', 'sale', 'corp', 'logi', 'stock', 'invest'],
            'Administración / Contabilidad': ['admi', 'acc', 'audi', 'mana', 'offi', 'hr', 'logi', 'reso', 'cont', 'huma', 'secre', 'plan']
        }
        for category, keywords in mapping.items():
            if any(k in m for k in keywords): return category
        return None

    async def train_from_files(self, filenames: list[str] = None):
        all_logs = []
        all_dfs = []
        try:
            for fname in filenames:
                path = os.path.join(self.datasets_dir, fname)
                if os.path.exists(path):
                    df = pd.read_csv(path, sep=None, engine='python', on_bad_lines='skip')
                    df_cleaned, logs = self.clean_data(df)
                    all_logs.extend(logs)
                    all_dfs.append(df_cleaned)

            full_df = pd.concat(all_dfs, ignore_index=True)
            X = full_df[['R', 'I', 'A', 'S', 'E', 'C']]
            y = full_df['Career_Category']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # THE "MOST PRECISE" MODEL: Random Forest
            rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
            rf_model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, rf_model.predict(X_test))
            
            # THE "EXPLAINABLE" MODEL: Decision Tree (for XAI visualization)
            tree_model = DecisionTreeClassifier(max_depth=12, random_state=42)
            tree_model.fit(X_train, y_train)
            
            joblib.dump(rf_model, self.rf_model_path)
            joblib.dump(tree_model, self.model_path)
            joblib.dump(['R', 'I', 'A', 'S', 'E', 'C'], self.features_path)
            
            stats = {
                "accuracy": float(accuracy),
                "n_samples": len(full_df),
                "trained_at": datetime.now().isoformat(),
                "classes": list(rf_model.classes_),
                "logs": all_logs
            }
            with open(self.stats_path, 'w') as f: json.dump(stats, f)
            return stats
        except Exception as e:
            print(traceback.format_exc())
            raise Exception(f"Training failed: {str(e)}")

    def explain_prediction(self, scores_dict: dict):
        if not os.path.exists(self.model_path): return None
        
        # Use RF for the final confidence/probabilities, but Tree for the path
        rf_model = joblib.load(self.rf_model_path)
        tree_model = joblib.load(self.model_path)
        features = joblib.load(self.features_path)
        X = np.array([[scores_dict.get(f, 0) for f in features]])
        
        # XAI: Use the Decision Tree to get the interpretable path
        node_indicator = tree_model.decision_path(X)
        leaf_id = tree_model.apply(X)[0]
        path = []
        node_indices = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
        for node_id in node_indices:
            if leaf_id == node_id:
                path.append({"node_id": int(node_id), "type": "leaf", "prediction": str(tree_model.classes_[np.argmax(tree_model.tree_.value[node_id])])})
            else:
                feature = features[tree_model.tree_.feature[node_id]]
                threshold = float(tree_model.tree_.threshold[node_id])
                val = float(X[0, tree_model.tree_.feature[node_id]])
                path.append({"node_id": int(node_id), "type": "decision", "feature": feature, "threshold": round(threshold, 2), "value": val})
        
        # ACCURACY: Use Random Forest for probabilities and final result
        probs = rf_model.predict_proba(X)[0]
        classes = rf_model.classes_
        sorted_indices = np.argsort(probs)[::-1]
        
        main_pred = str(classes[sorted_indices[0]])
        main_conf = float(probs[sorted_indices[0]])
        second_pred = str(classes[sorted_indices[1]])
        second_conf = float(probs[sorted_indices[1]])
        
        return {
            "decision_path": path,
            "full_tree": self.get_full_tree_structure(tree_model),
            "leaf_id": int(leaf_id),
            "insights": {
                "confidence": round(main_conf * 100, 2),
                "is_multipotential": (main_conf - second_conf) < 0.15,
                "second_option": {"career": second_pred, "confidence": round(second_conf * 100, 2)},
                "analysis": "Perfil con alta claridad vocacional" if (main_conf - second_conf) >= 0.15 else "Perfil multipotencial detectado."
            }
        }

    def get_full_tree_structure(self, model) -> Dict[str, Any]:
        tree = model.tree_
        features = joblib.load(self.features_path)
        def recurse(node: int) -> Dict[str, Any]:
            if tree.children_left[node] == tree.children_right[node]:
                return {"node_id": int(node), "type": "leaf", "prediction": str(model.classes_[np.argmax(tree.value[node])])}
            return {"node_id": int(node), "type": "decision", "feature": features[tree.feature[node]], "threshold": round(float(tree.threshold[node]), 2), "left": recurse(tree.children_left[node]), "right": recurse(tree.children_right[node])}
        return recurse(0)

    def get_model_stats(self):
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'r') as f: return json.load(f)
        return None

ml_service = MLService()
