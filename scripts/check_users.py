import asyncio
from sqlalchemy import select
from app.db.session import engine, AsyncSession
from app.db.models.user_model import User

async def check_users():
    async with AsyncSession(engine) as session:
        result = await session.execute(select(User))
        users = result.scalars().all()
        print("--- LISTA DE USUARIOS EN BD ---")
        if not users:
            print("No hay usuarios registrados.")
        for u in users:
            print(f"Email: {u.email} | Rol: {u.role} | Activo: {u.is_active}")
        print("-------------------------------")

if __name__ == "__main__":
    asyncio.run(check_users())
