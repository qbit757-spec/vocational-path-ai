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
        # 1. Detectar si es separado por Tabulaciones (como vimos en tu comando)
        if df.columns[0].count('\t') > 5:
            col_str = df.columns[0]
            df = pd.read_csv(os.path.join(self.datasets_dir, 'data.csv'), sep='\t', on_bad_lines='skip')
        
        df.columns = [str(c).strip() for c in df.columns]
        
        # 2. Extraer Features de Interés (RIASEC 48)
        riasec_cols = []
        for cat in ['R', 'I', 'A', 'S', 'E', 'C']:
            cols = [c for c in df.columns if c.startswith(cat) and c[1:].isdigit()]
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(3)
                riasec_cols.append(c)
        
        # 3. Extraer Features de Personalidad (TIPI 1-10)
        tipi_cols = [f'TIPI{i}' for i in range(1, 11) if f'TIPI{i}' in df.columns]
        for c in tipi_cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(4)
        
        # 4. Extraer Features de Inteligencia (VCL 1-16)
        vcl_cols = [f'VCL{i}' for i in range(1, 17) if f'VCL{i}' in df.columns]
        for c in vcl_cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
        # 5. Datos Demográficos Clave
        demo_cols = []
        for col in ['age', 'gender', 'education']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                demo_cols.append(col)
        
        # 6. Filtros de Calidad Científica
        if 'VCL6' in df.columns: df = df[df['VCL6'] == 0] # Filtro de honestidad
        if 'age' in df.columns: df = df[(df['age'] >= 14) & (df['age'] <= 80)]
        
        # Mapeo de Carrera
        if 'major' in df.columns:
            df['Career_Category'] = df['major'].apply(self._map_major_to_category)
            df = df.dropna(subset=['Career_Category'])
            # Balanceo de muestras
            df = df.groupby('Career_Category').apply(lambda x: x.sample(n=min(len(x), 10000), random_state=42)).reset_index(drop=True)
        
        all_features = riasec_cols + tipi_cols + vcl_cols + demo_cols
        return df, all_features

    def _map_major_to_category(self, major: Any) -> Optional[str]:
        if not isinstance(major, str): return None
        m = str(major).lower().strip()
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
        try:
            path = os.path.join(self.datasets_dir, filenames[0])
            df_raw = pd.read_csv(path, nrows=5) # Solo para detectar separador
            full_df, features = self.clean_data(df_raw) # Recargamos con el separador correcto internamente
            
            # Recargar el dataset completo de forma eficiente
            if full_df.empty:
                full_df = pd.read_csv(path, sep='\t' if '\t' in open(path).readline() else ',', on_bad_lines='skip')
                full_df, features = self.clean_data(full_df)

            X = full_df[features]
            y = full_df['Career_Category']
            y_codes = pd.Categorical(y)
            y_mapped = y_codes.codes
            
            X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.2, random_state=42)
            
            # XGBoost con máxima potencia
            model = XGBClassifier(n_estimators=800, max_depth=12, learning_rate=0.05, objective='multi:softprob', tree_method='hist', random_state=42)
            model.fit(X_train, y_train)
            
            accuracy = accuracy_score(y_test, model.predict(X_test))
            
            # Guardar activos
            joblib.dump(model, self.model_path)
            joblib.dump(features, self.features_path)
            joblib.dump(list(y_codes.categories), self.classes_path)
            
            stats = {"accuracy": float(accuracy), "n_samples": len(full_df), "features_used": len(features), "trained_at": datetime.now().isoformat()}
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
        
        # Preparar vector de entrada (rellenando con defaults si faltan)
        X_vec = []
        for f in features:
            default = 3 if f.startswith(('R','I','A','S','E','C')) else (4 if 'TIPI' in f else 0)
            X_vec.append(inputs.get(f, default))
        
        X_arr = np.array([X_vec])
        probs = model.predict_proba(X_arr)[0]
        idx = np.argsort(probs)[::-1]
        
        return {
            "insights": {
                "confidence": float(probs[idx[0]]),
                "is_multipotential": (probs[idx[0]] - probs[idx[1]]) < 0.10,
                "second_option": {"career": str(classes[idx[1]]), "confidence": round(probs[idx[1]] * 100, 2)},
                "prediction": str(classes[idx[0]])
            },
            "decision_path": [{"prediction": str(classes[idx[0]])}] # Simplificado para evitar colapso
        }

    def get_model_stats(self):
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'r') as f: return json.load(f)
        return None

ml_service = MLService()
