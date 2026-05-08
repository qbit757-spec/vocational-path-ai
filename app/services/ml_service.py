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
import traceback
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
        csv_files = [f for f in files if f.endswith('.csv')]
        return [{"filename": f, "size": os.path.getsize(os.path.join(self.datasets_dir, f))} for f in csv_files]

    def clean_data(self, df: pd.DataFrame):
        logs = []
        # Clean column names (strip whitespace/quotes)
        df.columns = [str(c).strip().replace('"', '') for c in df.columns]
        logs.append(f"Initial shape: {df.shape}")
        
        # Identify RIASEC columns dynamically
        # Some datasets use F instead of E for some items, we'll be flexible
        ria_map = {
            'R': [c for c in df.columns if c.startswith('R') and c[1:].isdigit()],
            'I': [c for c in df.columns if c.startswith('I') and c[1:].isdigit()],
            'A': [c for c in df.columns if c.startswith('A') and c[1:].isdigit()],
            'S': [c for c in df.columns if c.startswith('S') and c[1:].isdigit()],
            'E': [c for c in df.columns if (c.startswith('E') or c.startswith('F')) and c[1:].isdigit()],
            'C': [c for c in df.columns if c.startswith('C') and c[1:].isdigit()]
        }
        
        is_real_riasec = all(len(cols) >= 5 for cols in ria_map.values())
        
        if is_real_riasec:
            logs.append("Detected real-world RIASEC format with dynamic column matching...")
            
            # 1. VCL Validity Check
            vcl_cols = [c for c in df.columns if c.startswith('VCL')]
            if vcl_cols:
                before_vcl = len(df)
                # Filter by honesty (VCL6, 9, 12 should be 0 if they exist)
                for vc in ['VCL6', 'VCL9', 'VCL12']:
                    if vc in df.columns:
                        df[vc] = pd.to_numeric(df[vc], errors='coerce').fillna(0)
                        df = df[df[vc] == 0]
                logs.append(f"Dropped {before_vcl - len(df)} records (VCL check)")
            
            # 2. Age Filter
            if 'age' in df.columns:
                df['age'] = pd.to_numeric(df['age'], errors='coerce')
                df = df[(df['age'] >= 13) & (df['age'] <= 80)]
            
            # 3. Aggregate RIASEC items
            for cat, cols in ria_map.items():
                for c in cols:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(3)
                df[cat] = df[cols].mean(axis=1) * 2 # Scale to 0-10 (assuming 1-5 scale)
                # Adjustment: (Mean - 1) / (5 - 1) * 10
                df[cat] = ((df[cat]/2 - 1) / 4) * 10
            
            # 4. Map 'major' to 'Career_Category'
            if 'major' in df.columns:
                df['Career_Category'] = df['major'].apply(self._map_major_to_category)
                df = df.dropna(subset=['Career_Category'])
                logs.append(f"Mapped majors to categories. Final rows: {len(df)}")
            else:
                logs.append("WARNING: 'major' column not found, generating labels...")
                df['Career_Category'] = df.apply(self._synthetic_label, axis=1)
        else:
            logs.append("Format not recognized as 48-item. Looking for direct R,I,A,S,E,C columns.")
            required_cols = ['R', 'I', 'A', 'S', 'E', 'C', 'Career_Category']
            existing_cols = [c for c in required_cols if c in df.columns]
            df = df[existing_cols].dropna()
        
        return df, logs

    def _map_major_to_category(self, major: Any) -> Optional[str]:
        if not isinstance(major, str): return None
        major = str(major).lower().strip()
        if not major or major == 'nan': return None
        
        mapping = {
            'Ingeniería / Tecnología': ['eng', 'comp', 'tech', 'softw', 'civil', 'mech', 'it', 'math', 'physic', 'syst', 'science', 'data', 'web'],
            'Ciencias de la Salud': ['med', 'nurs', 'dent', 'bio', 'pharm', 'psych', 'health', 'vet', 'therap', 'medic', 'nurse'],
            'Artes y Diseño': ['art', 'design', 'music', 'dance', 'fashion', 'film', 'photo', 'paint', 'archit', 'literat', 'creat', 'write'],
            'Ciencias Sociales / Educación': ['edu', 'teach', 'soc', 'hist', 'law', 'polit', 'anthro', 'ling', 'phil', 'counsel', 'commun', 'social'],
            'Negocios / Derecho': ['bus', 'market', 'econ', 'law', 'finan', 'entre', 'trade', 'comm', 'business', 'legal', 'sales'],
            'Administración / Contabilidad': ['admin', 'acc', 'audit', 'manage', 'office', 'hr', 'logist', 'resource']
        }
        for category, keywords in mapping.items():
            if any(k in major for k in keywords):
                return category
        return None

    async def train_from_files(self, filenames: list[str] = None):
        all_logs = []
        all_dfs = []
        try:
            if not filenames:
                all_logs.append("No files provided. Using synthetic generator.")
                df = self.generate_synthetic_df(2000)
                all_dfs.append(df)
            else:
                for fname in filenames:
                    path = os.path.join(self.datasets_dir, fname)
                    if os.path.exists(path):
                        all_logs.append(f"Processing {fname}...")
                        try:
                            # sep=None with python engine handles tabs/commas/any delimiter automatically
                            df = pd.read_csv(path, sep=None, engine='python', on_bad_lines='skip')
                            df_cleaned, logs = self.clean_data(df)
                            all_logs.extend(logs)
                            if not df_cleaned.empty:
                                all_dfs.append(df_cleaned)
                        except Exception as e:
                            all_logs.append(f"Error in {fname}: {str(e)}")

            if not all_dfs:
                raise Exception("No valid data to train with.")

            full_df = pd.concat(all_dfs, ignore_index=True)
            X = full_df[['R', 'I', 'A', 'S', 'E', 'C']]
            y = full_df['Career_Category']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = DecisionTreeClassifier(max_depth=12, random_state=42)
            model.fit(X_train, y_train)
            
            accuracy = accuracy_score(y_test, model.predict(X_test))
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
        except Exception as e:
            print(traceback.format_exc())
            raise Exception(f"Training failed: {str(e)}")

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
        if not os.path.exists(self.model_path):
            return None
        model = joblib.load(self.model_path)
        features = joblib.load(self.features_path)
        X = np.array([[scores_dict.get(f, 0) for f in features]])
        node_indicator = model.decision_path(X)
        leaf_id = model.apply(X)[0]
        tree = model.tree_
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
                feature = features[tree.feature[node_id]]
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
        full_tree = self.get_full_tree_structure(model)
        class_probs = tree.value[leaf_id][0] / np.sum(tree.value[leaf_id][0])
        sorted_indices = np.argsort(class_probs)[::-1]
        main_prediction = str(model.classes_[sorted_indices[0]])
        main_confidence = float(class_probs[sorted_indices[0]])
        second_prediction = str(model.classes_[sorted_indices[1]]) if len(sorted_indices) > 1 else None
        second_confidence = float(class_probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0
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
            if tree.children_left[node] == tree.children_right[node]:
                return {
                    "node_id": int(node),
                    "type": "leaf",
                    "prediction": str(model.classes_[np.argmax(tree.value[node])])
                }
            else:
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
