import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import engine
from app.db.base import Base
# Import all models here for metadata discovery
from app.db.models.user_model import User
from app.db.models.test_model import VocationalTestResult

async def init_db():
    async with engine.begin() as conn:
        # Create tables
        print("Creating database tables...")
        await conn.run_sync(Base.metadata.create_all)
        print("Tables created successfully.")

if __name__ == "__main__":
    asyncio.run(init_db())
