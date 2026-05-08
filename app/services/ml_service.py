import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import json
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional

class MLService:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.ml_dir = os.path.join(self.base_dir, 'ml')
        self.model_dir = os.path.join(self.ml_dir, 'assets')
        self.datasets_dir = os.path.join(self.ml_dir, 'datasets')
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        
        self.model_path = os.path.join(self.model_dir, 'xgboost_model.joblib')
        self.tree_path = os.path.join(self.model_dir, 'decision_tree_model.joblib')
        self.features_path = os.path.join(self.model_dir, 'features.joblib')
        self.stats_path = os.path.join(self.model_dir, 'model_stats.json')
        self.classes_path = os.path.join(self.model_dir, 'classes.joblib')

    def save_dataset(self, filename: str, content: bytes):
        path = os.path.join(self.datasets_dir, filename)
        with open(path, "wb") as f: f.write(content)
        return path

    def list_datasets(self):
        if not os.path.exists(self.datasets_dir): return []
        return [{"filename": f, "size": os.path.getsize(os.path.join(self.datasets_dir, f))} for f in os.listdir(self.datasets_dir) if f.endswith('.csv')]

    def clean_data(self, df: pd.DataFrame):
        if df.columns[0].count('\t') > 5:
            df = pd.read_csv(os.path.join(self.datasets_dir, 'data.csv'), sep='\t', on_bad_lines='skip')
        
        df.columns = [str(c).strip() for c in df.columns]
        
        # RIASEC (48)
        riasec_cols = []
        for cat in ['R', 'I', 'A', 'S', 'E', 'C']:
            cols = [c for c in df.columns if c.startswith(cat) and c[1:].isdigit()]
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(3)
                riasec_cols.append(c)
        
        # TIPI + VCL
        extra_cols = [f'TIPI{i}' for i in range(1, 11)] + [f'VCL{i}' for i in range(1, 17)]
        for c in extra_cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
        # Demographics
        demo_cols = ['age', 'gender', 'education']
        for c in demo_cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
        # --- FILTRO 75% (ELIMINAR INDECISOS) ---
        for cat in ['R', 'I', 'A', 'S', 'E', 'C']:
            cols = [c for c in riasec_cols if c.startswith(cat)]
            df[f"score_{cat}"] = df[cols].mean(axis=1)
        
        df['score_std'] = df[[f"score_{cat}" for cat in ['R', 'I', 'A', 'S', 'E', 'C']]].std(axis=1)
        # Filtro estricto para perfiles puros
        df = df[df['score_std'] > 1.15]
        
        if 'major' in df.columns:
            df['Career_Category'] = df['major'].apply(self._map_major_to_category)
            df = df.dropna(subset=['Career_Category'])
            # Balanceo agresivo
            df = df.groupby('Career_Category').apply(lambda x: x.sample(n=min(len(x), 7000), random_state=42)).reset_index(drop=True)
        
        features = riasec_cols + [c for c in extra_cols if c in df.columns] + [c for c in demo_cols if c in df.columns]
        return df, features

    def _map_major_to_category(self, major: Any) -> Optional[str]:
        if not isinstance(major, str): return None
        m = str(major).lower().strip()
        # 4 CATEGORIAS MEGA-SOLIDAS PARA EL 75%
        mapping = {
            'Ingeniería y Tecnología': ['eng', 'comp', 'tech', 'soft', 'civil', 'mech', 'it', 'math', 'phys', 'syst', 'scie', 'data', 'web', 'electr', 'robot', 'mining', 'telecom', 'indust'],
            'Ciencias de la Salud': ['med', 'nurs', 'dent', 'bio', 'phar', 'psyc', 'heal', 'vet', 'thera', 'medic', 'nurse', 'doct', 'physio', 'biol', 'nutri', 'kine', 'obs'],
            'Negocios, Gestión y Derecho': ['bus', 'mark', 'econ', 'law', 'fina', 'entr', 'trad', 'comm', 'busi', 'lega', 'sale', 'corp', 'logi', 'stock', 'invest', 'admi', 'acc', 'audi', 'mana', 'offi', 'hr', 'logi', 'reso', 'cont', 'huma', 'secre', 'plan'],
            'Artes, Humanidades y Educación': ['art', 'desig', 'musi', 'danc', 'fash', 'film', 'phot', 'pain', 'lite', 'crea', 'writ', 'dram', 'thea', 'fine', 'graph', 'visu', 'animat', 'edu', 'teac', 'soc', 'hist', 'poli', 'anth', 'ling', 'phil', 'coun', 'comm', 'geog', 'inter', 'journa', 'sociol', 'human']
        }
        for category, keywords in mapping.items():
            if any(k in m for k in keywords): return category
        return None

    async def train_from_files(self, filenames: list[str] = None):
        try:
            path = os.path.join(self.datasets_dir, filenames[0])
            full_df = pd.read_csv(path, sep='\t' if '\t' in open(path).readline() else ',', on_bad_lines='skip')
            full_df, features = self.clean_data(full_df)

            X = full_df[features]
            y = full_df['Career_Category']
            y_codes = pd.Categorical(y)
            y_mapped = y_codes.codes
            
            X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.15, random_state=42)
            
            # XGBOOST ULTRA-PROFUNDO
            model = XGBClassifier(n_estimators=1200, max_depth=15, learning_rate=0.03, objective='multi:softprob', tree_method='hist', random_state=42)
            model.fit(X_train, y_train)
            
            accuracy = accuracy_score(y_test, model.predict(X_test))
            
            joblib.dump(model, self.model_path)
            joblib.dump(features, self.features_path)
            joblib.dump(list(y_codes.categories), self.classes_path)
            
            stats = {"accuracy": float(accuracy), "n_samples": len(full_df), "engine": "XGBoost-Ultra", "trained_at": datetime.now().isoformat()}
            with open(self.stats_path, 'w') as f: json.dump(stats, f)
            return stats
        except Exception as e:
            print(traceback.format_exc())
            raise Exception(f"Training failed: {str(e)}")

    def explain_prediction(self, inputs: dict):
        if not os.path.exists(self.model_path): return None
        model = joblib.load(self.model_path)
        features = joblib.load(self.features_path)
        classes = joblib.load(self.classes_path)
        
        X_vec = [inputs.get(f, (3 if f.startswith(('R','I','A','S','E','C')) else 0)) for f in features]
        probs = model.predict_proba(np.array([X_vec]))[0]
        idx = np.argsort(probs)[::-1]
        
        return {
            "insights": {
                "confidence": float(probs[idx[0]]),
                "is_multipotential": (probs[idx[0]] - probs[idx[1]]) < 0.10,
                "second_option": {"career": str(classes[idx[1]]), "confidence": round(probs[idx[1]] * 100, 2)},
                "prediction": str(classes[idx[0]])
            },
            "decision_path": [{"prediction": str(classes[idx[0]])}]
        }

    def get_model_stats(self):
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'r') as f: return json.load(f)
        return None

ml_service = MLService()
