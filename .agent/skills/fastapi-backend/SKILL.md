---
name: fastapi-backend
description: >
  Experto desarrollador de backends en Python con FastAPI. Usa este skill
  siempre que el usuario pida crear un backend, una API REST, microservicio,
  endpoints, autenticación, sistema con base de datos, o API para móvil/web.
  También aplica para tareas parciales como "agregar un endpoint", "crear el
  modelo X", "agregar autenticación JWT", o "dockerizar el proyecto".
  Si existe cualquier duda sobre si aplica, úsalo — este skill es agresivo.
  Genera backends listos para producción: modulares, seguros y escalables.
tags: [fastapi, backend, postgresql, async, jwt, alembic, docker, websocket, rate-limiting]
activation_rules:
  aggressive: true   # Ante cualquier duda de si aplica, activar
  must_use_database: postgresql+asyncpg
  require_pydantic_schemas: true
  require_rate_limit: true
  require_async: true
  auto_create_tables: true
---

# FastAPI Production Backend

Genera backends profesionales con FastAPI + PostgreSQL listos para producción.
El código debe ser modular, seguro y escalable desde el primer commit.

## Stack obligatorio

Usa siempre: FastAPI, PostgreSQL, SQLAlchemy Async, asyncpg, Pydantic v2,
Alembic, python-jose, passlib[bcrypt], slowapi, structlog, orjson, pytest, httpx, Docker.

Nunca uses: SQLite, MySQL, MongoDB, ORMs síncronos, otros frameworks.

# Objetivo del Skill

Generar backends profesionales utilizando:

- FastAPI
- PostgreSQL
- SQLAlchemy Async
- JWT Authentication
- Arquitectura modular
- Seguridad integrada
- Documentación automática
- CRUD completo de todos los endpoints (Post, Get, Put, Delete, Patch)
- Primer usuario registrado automaticamente es el administrador con rol admin
- Docker
- Migraciones
- Observabilidad

El backend generado debe ser **escalable, mantenible y seguro**.

---

# Stack Tecnológico Obligatorio

Siempre utilizar:

- FastAPI
- PostgreSQL
- SQLAlchemy Async
- asyncpg
- Pydantic
- Alembic
- Docker
- httpx

Nunca utilizar:

- SQLite
- MySQL
- MongoDB
- ORM síncronos
- frameworks distintos a FastAPI

---

# Arquitectura del Proyecto

El backend debe generarse con la siguiente estructura obligatoria en la raiz del proyecto.

```
app/

    main.py

    core/
        config.py
        security.py
        middleware.py
        logging.py
        rate_limiter.py
        events.py

    db/
        base.py
        session.py
        init_db.py
        migrations/

        models/

    schemas/
        auth_schema.py
        user_schema.py

    repositories/
        user_repo.py

    services/
        user_service.py
        auth_service.py

    api/
        deps.py
        router.py

        v1/
            auth.py
            users.py

    websocket/
        manager.py

    tasks/
        background_tasks.py

    events/
        event_bus.py
        handlers/


docs/
    api/

scripts/

```

---

# requirements.txt optimizado base

algunos requerimientos que suelen ser obligatorios: 

fastapi==0.111.0
uvicorn[standard]==0.29.0
gunicorn==22.0.0
pydantic==2.7.1
pydantic-settings==2.2.1
sqlalchemy[asyncio]==2.0.30
asyncpg==0.29.0
alembic==1.13.1
python-jose[cryptography]==3.3.0
bcrypt==4.0.1
passlib[bcrypt]==1.7.4
python-multipart==0.0.9
email-validator==2.0.0
httpx==0.27.0
slowapi==0.1.9
python-dotenv==1.0.1
structlog==24.1.0
orjson==3.10.3
tenacity==8.2.3

---

# Convenciones de Nombres

Cada entidad debe tener:

```
{entity}_model.py
{entity}_schema.py
{entity}_repo.py
{entity}_service.py
```

Ejemplo:

```
user_model.py
user_schema.py
user_repo.py
user_service.py
```

---

# Base de Datos

La base de datos **siempre debe ser PostgreSQL**.

URL estándar:

```
postgresql+asyncpg://user:password@localhost/dbname
```

---

# Inicialización Automática de Base de Datos

Siempre crear `init_db.py`.

Debe ejecutarse en startup.

Ejemplo:

```python
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

Esto evita que el backend falle si las tablas no existen.

---

# Modelos de Base de Datos

Todos los modelos deben incluir:

- id
- created_at
- updated_at
- soft delete
- índices

Ejemplo:

```python
class User(Base):

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    email = Column(String, unique=True, index=True)

    password_hash = Column(String)

    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime)
    updated_at = Column(DateTime)
```

---

# Schemas Pydantic

Nunca usar:

```
dict
```

Siempre usar schemas.

Ejemplo:

```python
class UserCreate(BaseModel):

    email: EmailStr
    password: str
```

---

# Endpoints REST Estándar

Todos los recursos deben seguir:

```
GET     /api/v1/{entity}
GET     /api/v1/{entity}/{id}
POST    /api/v1/{entity}
PUT     /api/v1/{entity}/{id}
DELETE  /api/v1/{entity}/{id}
```

---

# Paginación

Todos los endpoints de lista deben soportar:

```
skip
limit
```

Ejemplo:

```
GET /users?skip=0&limit=20
```

---

# Autenticación

Debe implementarse:

- JWT access token
- refresh token (debe generarse un endpoint para generar un nuevo access token)
- password hashing

Algoritmo:

```
HS256
```

Password hashing:

```
bcrypt
```

Endpoints obligatorios:

```
POST /auth/register
POST /auth/login
POST /auth/refresh
```

---

# Seguridad

Siempre implementar:

- password hashing
- JWT
- rate limiting
- validación de datos para evitar inyección SQL

Ejemplo rate limit:

```
login: 5/min
register: 3/min
```

---

# Middleware Obligatorio

Siempre incluir:

- CORS
- logging
- TrustedHostMiddleware
- ProxyHeadersMiddleware

Nota : 
el nombre del parámetro para el middleware de cabeceras de proxy es trusted_hosts (no trusted_proxies)

---

# Event Bus Interno

Para desacoplar lógica.

Ejemplo:

```
UserCreatedEvent
```

Handlers:

```
send_email
create_profile
log_event
```

---

# Background Tasks

Usar para:

- envío de emails
- procesamiento en segundo plano
- logs
- notificaciones

Ejemplo:

```
background_tasks.add_task(send_email)
```

---

# WebSockets (Opcional)

Si el sistema requiere realtime.

Ejemplo:

```
/ws/{channel}
```

---

# Redis Cache (Opcional)

Usar para:

- cache
- rate limiting
- sesiones

---

# Documentación de API

Cada endpoint debe tener documentación en:

```
docs/api/
```

Archivo ejemplo:

```
docs/api/users.md
```

Debe incluir:

- descripción
- request example
- response example
- curl

---

# Docker

Siempre generar:

```
Dockerfile
docker-compose.yml
```

Servicios mínimos:

```
backend
postgres
```

---

# Migraciones

Siempre usar Alembic.

Comandos:

```
alembic revision --autogenerate
alembic upgrade head
```

---

# Observabilidad

Implementar:

- logging estructurado
- métricas opcionales
- manejo de errores

---

# Control de Errores

Todas las APIs deben devolver:

```
status_code
error_message
error_code
```

Ejemplo:

```
{
  "error": "USER_NOT_FOUND"
}
```

---

# Checklist Final

Antes de finalizar el backend verificar:

- PostgreSQL configurado
- CRUD completo (post, get, put, delete, patch) en todos los modelos
- SQLAlchemy Async
- init_db ejecutado
- JWT implementado
- password hashing
- rate limiting
- Docker configurado
- Alembic configurado
- documentación generada
- middleware configurado
- endpoints REST estándar
- schemas creados
- repositories creados
- services creados
- refresh token
- volumenes para persistencia en docker

Si alguno falta el backend **NO está completo**.

---

# mejora al arrancar el proyecto

- Al arrancar el proyecto se debe ejecutar el script `init_db.py` para crear las tablas en la base de datos.
- El script `init_db.py` se encuentra en la carpeta `app/db/`.
- El script `init_db.py` se ejecuta automáticamente al arrancar el backend.
- El backend intenta conectarse a la base de datos antes de que PostgreSQL esté listo para recibir conexiones, a pesar de que el contenedor de la DB ya haya "arrancado".
Para solucionar esto, aplicaremos una estrategia de reintentos (retries) usando tenacity
- En aplicaciones de producción, lo ideal es que tanto Python como la base de datos manejen siempre zonas horarias (UTC por defecto).