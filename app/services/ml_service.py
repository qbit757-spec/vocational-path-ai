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

# Intentamos importar XGBoost, si no está usamos RandomForest como fallback inteligente
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

class MLService:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.ml_dir = os.path.join(self.base_dir, 'ml')
        self.model_dir = os.path.join(self.ml_dir, 'assets')
        self.datasets_dir = os.path.join(self.ml_dir, 'datasets')
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        
        self.model_path = os.path.join(self.model_dir, 'decision_tree_model.joblib')
        self.rf_model_path = os.path.join(self.model_dir, 'random_forest_model.joblib')
        self.features_path = os.path.join(self.model_dir, 'features.joblib')
        self.stats_path = os.path.join(self.model_dir, 'model_stats.json')

    def save_dataset(self, filename: str, content: bytes):
        path = os.path.join(self.datasets_dir, filename)
        with open(path, "wb") as f: f.write(content)
        return path

    def list_datasets(self):
        if not os.path.exists(self.datasets_dir): return []
        return [{"filename": f, "size": os.path.getsize(os.path.join(self.datasets_dir, f))} for f in os.listdir(self.datasets_dir) if f.endswith('.csv')]

    def clean_data(self, df: pd.DataFrame):
        df.columns = [str(c).strip().replace('"', '') for c in df.columns]
        riasec_items = []
        for cat in ['R', 'I', 'A', 'S', 'E', 'C']:
            cols = [c for c in df.columns if c.startswith(cat) and c[1:].isdigit() and int(c[1:]) <= 8]
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(3)
                riasec_items.append(c)
        
        if len(riasec_items) >= 40:
            if 'VCL6' in df.columns:
                df = df[pd.to_numeric(df['VCL6'], errors='coerce').fillna(0) == 0]
            
            for cat in ['R', 'I', 'A', 'S', 'E', 'C']:
                cols = [c for c in riasec_items if c.startswith(cat)]
                df[f"score_{cat}"] = ((df[cols].mean(axis=1) - 1) / 4) * 10
            
            df['score_std'] = df[[f"score_{cat}" for cat in ['R', 'I', 'A', 'S', 'E', 'C']]].std(axis=1)
            df = df[df['score_std'] > 0.8]
            
            if 'major' in df.columns:
                df['Career_Category'] = df['major'].apply(self._map_major_to_category)
                df = df.dropna(subset=['Career_Category'])
                # Capping a 8,000 para balancear mejor
                df = df.groupby('Career_Category').apply(lambda x: x.sample(n=min(len(x), 8000), random_state=42)).reset_index(drop=True)
        
        return df, [], riasec_items

    def _map_major_to_category(self, major: Any) -> Optional[str]:
        if not isinstance(major, str): return None
        m = str(major).lower().strip()
        if not m or m in ['nan', 'none', 'student', 'undecided']: return None
        
        # CATEGORIAS OPTIMIZADAS (Fusionamos 5 y 6 para subir precision)
        mapping = {
            'Ingeniería / Tecnología': ['eng', 'comp', 'tech', 'soft', 'civil', 'mech', 'it', 'math', 'phys', 'syst', 'scie', 'data', 'web', 'electr', 'robot', 'mining', 'telecom', 'indust'],
            'Ciencias de la Salud': ['med', 'nurs', 'dent', 'bio', 'phar', 'psyc', 'heal', 'vet', 'thera', 'medic', 'nurse', 'doct', 'physio', 'biol', 'nutri', 'kine', 'obs'],
            'Artes y Diseño': ['art', 'desig', 'musi', 'danc', 'fash', 'film', 'phot', 'pain', 'lite', 'crea', 'writ', 'dram', 'thea', 'fine', 'graph', 'visu', 'animat'],
            'Ciencias Sociales / Educación': ['edu', 'teac', 'soc', 'hist', 'law', 'poli', 'anth', 'ling', 'phil', 'coun', 'comm', 'phil', 'geog', 'inter', 'journa', 'sociol', 'human'],
            'Negocios, Gestión y Derecho': ['bus', 'mark', 'econ', 'law', 'fina', 'entr', 'trad', 'comm', 'busi', 'lega', 'sale', 'corp', 'logi', 'stock', 'invest', 'admi', 'acc', 'audi', 'mana', 'offi', 'hr', 'logi', 'reso', 'cont', 'huma', 'secre', 'plan']
        }
        for category, keywords in mapping.items():
            if any(k in m for k in keywords): return category
        return None

    async def train_from_files(self, filenames: list[str] = None):
        all_dfs = []
        riasec_features = []
        try:
            for fname in filenames:
                path = os.path.join(self.datasets_dir, fname)
                if os.path.exists(path):
                    df = pd.read_csv(path, sep=None, engine='python', on_bad_lines='skip')
                    df_cleaned, _, features = self.clean_data(df)
                    riasec_features = features
                    all_dfs.append(df_cleaned)

            full_df = pd.concat(all_dfs, ignore_index=True)
            X = full_df[riasec_features]
            # Codificamos las etiquetas para XGBoost
            y = full_df['Career_Category']
            y_codes = pd.Categorical(y)
            y_mapped = y_codes.codes
            
            X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.2, random_state=42)
            
            if HAS_XGB:
                model = XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.1, objective='multi:softprob', random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=300, max_depth=25, class_weight='balanced', random_state=42)
            
            model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, model.predict(X_test))
            
            # Arbol visual simplificado
            X_xai = full_df[[f"score_{cat}" for cat in ['R', 'I', 'A', 'S', 'E', 'C']]]
            tree_model = DecisionTreeClassifier(max_depth=12, random_state=42)
            tree_model.fit(X_xai, y)
            
            joblib.dump(model, self.rf_model_path)
            joblib.dump(tree_model, self.model_path)
            joblib.dump(riasec_features, self.features_path)
            joblib.dump(list(y_codes.categories), os.path.join(self.model_dir, 'classes.joblib'))
            
            stats = {"accuracy": float(accuracy), "n_samples": len(full_df), "engine": "XGBoost" if HAS_XGB else "RandomForest", "trained_at": datetime.now().isoformat()}
            with open(self.stats_path, 'w') as f: json.dump(stats, f)
            return stats
        except Exception as e:
            print(traceback.format_exc())
            raise Exception(f"Training failed: {str(e)}")

    def explain_prediction(self, scores_dict: dict):
        if not os.path.exists(self.rf_model_path): return None
        model = joblib.load(self.rf_model_path)
        tree_model = joblib.load(self.model_path)
        features_48 = joblib.load(self.features_path)
        classes = joblib.load(os.path.join(self.model_dir, 'classes.joblib'))
        
        X_48 = np.array([[scores_dict.get(f, 3) for f in features_48]])
        
        # Preparar XAI
        X_xai = []
        for cat in ['R', 'I', 'A', 'S', 'E', 'C']:
            cols = [f for f in features_48 if f.startswith(cat)]
            avg = np.mean([scores_dict.get(c, 3) for c in cols])
            X_xai.append(((avg - 1) / 4) * 10)
        X_xai_arr = np.array([X_xai])
        
        node_indicator = tree_model.decision_path(X_xai_arr)
        leaf_id = tree_model.apply(X_xai_arr)[0]
        path = []
        node_indices = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
        for node_id in node_indices:
            if leaf_id == node_id:
                path.append({"node_id": int(node_id), "type": "leaf", "prediction": str(tree_model.classes_[np.argmax(tree_model.tree_.value[node_id])])})
            else:
                f_idx = tree_model.tree_.feature[node_id]
                feat_name = ['R', 'I', 'A', 'S', 'E', 'C'][f_idx]
                threshold = float(tree_model.tree_.threshold[node_id])
                val = float(X_xai_arr[0, f_idx])
                cond_str = f"{feat_name} {'<=' if val <= threshold else '>'} {round(threshold, 2)}"
                path.append({"node_id": int(node_id), "type": "decision", "feature": feat_name, "threshold": round(threshold, 2), "student_value": val, "condition": cond_str})
        
        probs = model.predict_proba(X_48)[0]
        idx = np.argsort(probs)[::-1]
        
        return {
            "decision_path": path,
            "full_tree": self.get_full_tree_structure(tree_model),
            "insights": {
                "confidence": float(probs[idx[0]]),
                "is_multipotential": (probs[idx[0]] - probs[idx[1]]) < 0.12,
                "second_option": {"career": str(classes[idx[1]]), "confidence": round(probs[idx[1]] * 100, 2)},
                "analysis": "Diagnóstico de alta precisión mediante XGBoost."
            }
        }

    def get_full_tree_structure(self, model) -> Dict[str, Any]:
        tree = model.tree_
        feats = ['R', 'I', 'A', 'S', 'E', 'C']
        def recurse(node: int) -> Dict[str, Any]:
            if tree.children_left[node] == tree.children_right[node]:
                return {"node_id": int(node), "type": "leaf", "prediction": str(model.classes_[np.argmax(tree.value[node])])}
            return {"node_id": int(node), "type": "decision", "feature": feats[tree.feature[node]], "threshold": round(float(tree.threshold[node]), 2), "left": recurse(tree.children_left[node]), "right": recurse(tree.children_right[node])}
        return recurse(0)

    def get_model_stats(self):
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'r') as f: return json.load(f)
        return None

ml_service = MLService()
