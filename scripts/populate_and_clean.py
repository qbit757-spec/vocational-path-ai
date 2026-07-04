import asyncio
import httpx
import random
import sys
import os
from typing import List, Dict, Any

# Configure stdout to support UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Configure API URL (can be set via env var API_BASE_URL, default is the live server)
API_BASE_URL = os.environ.get("API_BASE_URL", "https://apitesis.fanecorp.com")
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@orientatufuturo.pe")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

# List of typical Peruvian names and surnames for UGEL 03 students
NOMBRES = ["Luis", "Jose", "Maria", "Sofia", "Carlos", "Juan", "Ana", "Diego", "Mateo", "Camila", 
           "Lucia", "Jorge", "David", "Daniel", "Andrea", "Valeria", "Sebastian", "Alejandro", 
           "Gabriel", "Rosa", "Carmen", "Miguel", "Angel", "Manuel", "Gabriela", "Mariana", 
           "Santiago", "Nicolas", "Paula", "Adriana", "Fabrizio", "Alvaro", "Daniela", "Camilo"]
APELLIDOS = ["Quispe", "Flores", "Sanchez", "Garcia", "Rodriguez", "Huaman", "Rojas", "Vasquez", 
             "Mamani", "Torres", "Lopez", "Gutierrez", "Diaz", "Chavez", "Gomez", "Mendoza", 
             "Castillo", "Villanueva", "Fernandez", "Alvarez", "Ramos", "Espinoza", "Salazar", 
             "Cruz", "Silva", "Ortiz", "Perez", "Arias", "Campos", "Caceres", "Medina", "Rios"]

def generate_random_name() -> str:
    return f"{random.choice(NOMBRES)} {random.choice(APELLIDOS)} {random.choice(APELLIDOS)}"

def generate_test_answers(career_type: str) -> List[Dict[str, int]]:
    """
    Generates answers from 1 to 5 for the 48 questions.
    Forces the targeted categories to be high (4-5) and others to be low (1-2)
    to guarantee the desired Decision Tree prediction.
    """
    answers = []
    
    # Career Category mappings:
    # 1. Ingeniería y Tecnología: Realista (1-8) & Investigativo (9-16)
    # 2. Ciencias de la Salud: Social (25-32) & Investigativo (9-16)
    # 3. Artes, Humanidades y Educación: Artístico (17-24) & Social (25-32)
    # 4. Negocios, Gestión y Derecho: Emprendedor (33-40) & Convencional (41-48)
    
    high_ranges = []
    if career_type == "Ingeniería y Tecnología":
        high_ranges = [(1, 8), (9, 16)]
    elif career_type == "Ciencias de la Salud":
        high_ranges = [(25, 32), (9, 16)]
    elif career_type == "Artes, Humanidades y Educación":
        high_ranges = [(17, 24), (25, 32)]
    elif career_type == "Negocios, Gestión y Derecho":
        high_ranges = [(33, 40), (41, 48)]
        
    for q_id in range(1, 49):
        is_high = any(start <= q_id <= end for start, end in high_ranges)
        if is_high:
            val = random.randint(4, 5)
        else:
            val = random.randint(1, 2)
        answers.append({"question_id": q_id, "value": val})
        
    return answers

async def clean_database_directly() -> bool:
    """
    Attempts to clean student tables directly using SQLAlchemy if the DB is running locally or accessible.
    """
    try:
        from app.db.session import AsyncSessionLocal
        from app.db.models.user_model import User
        from app.db.models.test_model import VocationalTestResult
        from sqlalchemy import delete
        
        async with AsyncSessionLocal() as session:
            # Delete test results
            await session.execute(delete(VocationalTestResult))
            # Delete student users
            await session.execute(delete(User).where(User.role != "admin"))
            await session.commit()
            print("Direct DB: Limpieza realizada con éxito mediante conexión directa a la Base de Datos.")
            return True
    except Exception as e:
        print(f"Direct DB: No se pudo conectar a la base de datos directamente o falló ({e}).")
        return False

async def process_student(client: httpx.AsyncClient, student_index: int, career_type: str) -> Dict[str, Any]:
    full_name = generate_random_name()
    # Unique email using index
    email = f"estudiante_{student_index}@ugel03.edu.pe"
    password = f"Estudiante{student_index}!"
    
    # Random age between 14 and 16
    age = random.randint(14, 16)
    # Random gender (1: Male, 2: Female)
    gender = random.randint(1, 2)
    # Random education (2: 3rd grade, 3: 4th grade of secondary)
    education = random.randint(2, 3)
    
    try:
        # 1. Register student
        reg_res = await client.post(
            f"{API_BASE_URL}/api/v1/auth/register",
            json={
                "email": email,
                "password": password,
                "full_name": full_name,
                "is_active": True
            }
        )
        if reg_res.status_code != 200:
            return {"status": "error", "email": email, "step": "register", "detail": reg_res.text}
            
        # 2. Login student
        login_res = await client.post(
            f"{API_BASE_URL}/api/v1/auth/login",
            data={
                "username": email,
                "password": password
            }
        )
        if login_res.status_code != 200:
            return {"status": "error", "email": email, "step": "login", "detail": login_res.text}
            
        token_data = login_res.json()
        token = token_data["access_token"]
        
        # 3. Submit test answers
        answers = generate_test_answers(career_type)
        headers = {"Authorization": f"Bearer {token}"}
        submit_res = await client.post(
            f"{API_BASE_URL}/api/v1/test/submit",
            json={
                "answers": answers,
                "age": age,
                "gender": gender,
                "education": education
            },
            headers=headers
        )
        if submit_res.status_code != 200:
            return {"status": "error", "email": email, "step": "submit", "detail": submit_res.text}
            
        res_data = submit_res.json()
        return {
            "status": "success", 
            "name": full_name, 
            "email": email, 
            "age": age,
            "grade": "3ro" if education == 2 else "4to",
            "recommendation": res_data.get("recommendation")
        }
    except Exception as e:
        return {"status": "error", "email": email, "step": "exception", "detail": str(e)}

async def main():
    print("=" * 60)
    print(f"SISTEMA DE POBLACIÓN DE ESTUDIANTES - UGEL 03 LIMA CENTRO")
    print(f"URL de la API: {API_BASE_URL}")
    print("=" * 60)
    
    # 1. Try direct database cleanup first (useful if run on the server/localhost)
    print("Intentando limpieza directa de base de datos...")
    db_cleaned = await clean_database_directly()
    
    # 2. Login as Admin
    print("Iniciando sesión como Administrador en la API...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            admin_login = await client.post(
                f"{API_BASE_URL}/api/v1/auth/login",
                data={"username": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
            )
            if admin_login.status_code != 200:
                print(f"Error al iniciar sesión como administrador: {admin_login.text}")
                # If we couldn't log in but we already cleaned the DB directly and we are targeting localhost, we can still proceed
                if not db_cleaned:
                    sys.exit(1)
            else:
                admin_token = admin_login.json()["access_token"]
                admin_headers = {"Authorization": f"Bearer {admin_token}"}
                print("Sesión de administrador en la API iniciada correctamente.")
                
                # If DB was not cleaned directly, clean it via the API
                if not db_cleaned:
                    print("Limpiando base de datos de estudiantes mediante API Admin...")
                    reset_res = await client.delete(
                        f"{API_BASE_URL}/api/v1/admin/reset-students",
                        headers=admin_headers
                    )
                    if reset_res.status_code != 200:
                        print(f"Error al resetear la base de datos de estudiantes mediante la API: {reset_res.text}")
                        print("Continuando con la población...")
                    else:
                        print("Base de datos limpia de estudiantes mediante la API.")
                
        except Exception as e:
            print(f"Error durante la fase de administrador: {e}")
            if not db_cleaned:
                sys.exit(1)
            
        # 3. Define target distribution for 59 students (uneven, realistic)
        # Total = 59 students
        # Ingeniería y Tecnología: 22
        # Negocios, Gestión y Derecho: 16
        # Ciencias de la Salud: 12
        # Artes, Humanidades y Educación: 9
        targets = []
        targets.extend(["Ingeniería y Tecnología"] * 22)
        targets.extend(["Negocios, Gestión y Derecho"] * 16)
        targets.extend(["Ciencias de la Salud"] * 12)
        targets.extend(["Artes, Humanidades y Educación"] * 9)
        
        # Shuffle to mix them up
        random.shuffle(targets)
        
        print(f"\nIniciando registro y envío de tests para 59 estudiantes...")
        
        tasks = []
        for i, career in enumerate(targets, start=1):
            tasks.append(process_student(client, i, career))
            
        results = await asyncio.gather(*tasks)
        
        # 4. Process results and print summary
        success_count = 0
        error_count = 0
        career_counts = {}
        
        print("\n" + "-" * 60)
        print("RESUMEN DE ESTUDIANTES CREADOS:")
        print("-" * 60)
        
        for res in results:
            if res["status"] == "success":
                success_count += 1
                rec = res["recommendation"]
                career_counts[rec] = career_counts.get(rec, 0) + 1
                if success_count <= 10 or success_count in [20, 30, 40, 50, 59]:
                    print(f"Estudiante {success_count}: {res['name']} ({res['email']}) | Edad: {res['age']} | Grado: {res['grade']} | Carrera: {rec}")
            else:
                error_count += 1
                print(f"ERROR en {res['email']} ({res['step']}): {res['detail']}")
                
        print("=" * 60)
        print(f"Total exitosos: {success_count} / 59")
        print(f"Total con error: {error_count}")
        print("\nDistribución por carreras recomendadas:")
        for car, count in career_counts.items():
            print(f" - {car}: {count} estudiantes")
        print("=" * 60)

if __name__ == "__main__":
    if sys.platform == 'win32':
        # Evitar RuntimeError en Windows con asyncio en bucle cerrado
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
