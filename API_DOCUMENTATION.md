# 📘 Documentación de la API - Sistema de Elección de Carreras (UGEL 03)

Esta API proporciona el motor de inteligencia artificial y la gestión de datos para el Sistema de Orientación Vocacional.

**Base URL (Producción):** `https://apitesis.fanecorp.com/api/v1`  
**Base URL (Local):** `http://localhost:8098/api/v1`

---

## 🔐 Autenticación (`auth`)

Gestión de usuarios y acceso mediante tokens JWT.

### 1. Registrar Usuario
Crea una nueva cuenta de estudiante o administrador.

*   **Endpoint:** `POST /auth/register`
*   **Body (JSON):**
    ```json
    {
      "email": "estudiante@ejemplo.com",
      "password": "clave_segura",
      "full_name": "Nombre Completo"
    }
    ```
*   **Uso:** Llamar al inicio para que el estudiante pueda guardar sus resultados.

### 2. Iniciar Sesión (Login)
Obtiene un token de acceso para realizar el test.

*   **Endpoint:** `POST /auth/login`
*   **Body (Form Data):**
    *   `username`: email
    *   `password`: contraseña
*   **Respuesta:**
    ```json
    {
      "access_token": "eyJhbG...",
      "token_type": "bearer"
    }
    ```
*   **Uso:** El token recibido debe enviarse en la cabecera `Authorization: Bearer <token>` en las llamadas del test.

---

## 📝 Test Vocacional (`vocational-test`)

Lógica principal del test basado en RIASEC.

### 3. Enviar Test
Procesa las respuestas y devuelve una recomendación de carrera generada por la IA.

*   **Endpoint:** `POST /test/submit`
*   **Headers:** `Authorization: Bearer <token>`
*   **Body (JSON):**
    ```json
    {
      "scores": {
        "R": 8, "I": 10, "A": 5, "S": 7, "E": 4, "C": 3
      }
    }
    ```
*   **Respuesta:**
    ```json
    {
      "recommendation": "Ingeniería / Tecnología",
      "details": "Tu perfil muestra alta afinidad con el razonamiento técnico...",
      "created_at": "2026-05-06..."
    }
    ```

### 4. Obtener Historial
Recupera todos los tests realizados por el usuario autenticado.

*   **Endpoint:** `GET /test/history`
*   **Headers:** `Authorization: Bearer <token>`
*   **Uso:** Para mostrar al estudiante sus progresos o resultados anteriores.

---

## 🤖 Machine Learning (`machine-learning`)

Panel de control para la administración del modelo de IA.

### 5. Ver Estadísticas (`stats`)
Muestra el estado actual del modelo de Árbol de Decisión.

*   **Endpoint:** `GET /ml/stats`
*   **Respuesta:** Incluye `accuracy` (precisión), fecha de entrenamiento y logs de limpieza.
*   **Uso:** Para monitorear la salud del modelo desde el panel Admin.

### 6. Subir Dataset (`upload-dataset`)
Carga archivos CSV con datos reales para el entrenamiento.

*   **Endpoint:** `POST /ml/upload-dataset`
*   **Body (Multipart):** Archivo `.csv`
*   **Uso:** El administrador sube nuevos datos recolectados de campo.

### 7. Listar Datasets (`datasets`)
Muestra qué archivos están disponibles en el servidor para entrenar.

*   **Endpoint:** `GET /ml/datasets`

### 8. Entrenar Modelo (`train`)
Dispara el proceso de "estudio" de la IA sobre los datasets seleccionados.

*   **Endpoint:** `POST /ml/train`
*   **Body (JSON):**
    ```json
    ["dataset_lima.csv", "dataset_provincias.csv"]
    ```
*   **Uso:** Si se envía vacío `[]`, el modelo se entrena con datos sintéticos base.

---

## 🌐 General

### 9. Root
Verificación de estado del servidor.

*   **Endpoint:** `GET /`
*   **Respuesta:** `{"message": "Welcome to the Vocational Guidance API"}`

---

## 🛠️ Notas Técnicas
- **CORS:** La API acepta peticiones desde `localhost:3000` y `vercel.app` (configurado).
- **Seguridad:** Los passwords se encriptan con `PBKDF2` para máxima compatibilidad y seguridad en servidores modernos.
- **Base de Datos:** PostgreSQL 15, aislada del exterior por seguridad.
