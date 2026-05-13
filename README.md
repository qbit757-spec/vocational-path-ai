# Sistema web para mejorar el proceso de elección de carreras profesionales - UGEL 03

Sistema de Orientación Vocacional basado en Inteligencia Artificial para estudiantes de secundaria, utilizando el modelo RIASEC (Holland Codes) y algoritmos de Árboles de Decisión.

## 🚀 Características del Backend

- **API FastAPI**: Backend modular y rápido.
- **Modelo de IA**: Clasificador basado en Decision Trees para recomendaciones precisas.
- **Gestión de Datasets**: Carga y limpieza de datos sintéticos y reales desde el panel de administración.
- **Autenticación**: Sistema de registro y login seguro.
- **Base de Datos**: Integración con PostgreSQL (soporta Docker).

## 🛠️ Requisitos

- Python 3.10+
- PostgreSQL
- Docker (Opcional)

## 💻 Instalación Local

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. Configurar variables de entorno en `.env`:
   ```env
   POSTGRES_SERVER=localhost
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=admin
   POSTGRES_DB=vocational_db
   SECRET_KEY=tu_clave_secreta
   ```

3. Ejecutar el servidor:
   ```bash
   uvicorn app.main:app --reload
   ```

## 🧪 Entrenamiento del Modelo

Puedes entrenar el modelo desde la terminal o vía API:
```bash
set PYTHONPATH=.
python scripts/train_model.py
```

---
Desarrollado para la mejora del proceso de elección de carreras profesionales en estudiantes de 3.º y 4.º de secundaria en Lima Centro (UGEL 03).
