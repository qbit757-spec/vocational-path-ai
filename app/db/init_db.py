import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import engine
from app.db.base import Base
# Import all models here for metadata discovery
from app.db.models.user_model import User
from app.db.models.question_model import Question

async def init_db():
    async with engine.begin() as conn:
        print("Creating database tables...")
        await conn.run_sync(Base.metadata.create_all)
        print("Tables created successfully.")
    
    # Seeding initial questions
    async with AsyncSession(engine) as session:
        from sqlalchemy import select
        result = await session.execute(select(Question))
        if not result.scalars().first():
            print("Seeding initial questions...")
            initial_questions = [
                # Realistic
                {"text": "¿Te gustaría trabajar reparando aparatos electrónicos o motores?", "category": "R"},
                {"text": "¿Te interesa el manejo de herramientas y maquinaria pesada?", "category": "R"},
                # Investigative
                {"text": "¿Te apasiona investigar el porqué de los fenómenos naturales?", "category": "I"},
                {"text": "¿Disfrutas resolviendo problemas matemáticos complejos?", "category": "I"},
                # Artistic
                {"text": "¿Te gusta expresar tus ideas a través del dibujo o la pintura?", "category": "A"},
                {"text": "¿Te gustaría escribir cuentos o poemas?", "category": "A"},
                # Social
                {"text": "¿Te sientes bien ayudando a personas con problemas personales?", "category": "S"},
                {"text": "¿Te gustaría trabajar como profesor o instructor?", "category": "S"},
                # Enterprising
                {"text": "¿Te gustaría dirigir tu propia empresa o negocio?", "category": "E"},
                {"text": "¿Te sientes cómodo hablando en público para convencer a otros?", "category": "E"},
                # Conventional
                {"text": "¿Te gusta mantener tus cosas perfectamente ordenadas y clasificadas?", "category": "C"},
                {"text": "¿Te gustaría trabajar con hojas de cálculo y presupuestos?", "category": "C"},
            ]
            for q in initial_questions:
                session.add(Question(**q))
            await session.commit()
            print("Initial questions seeded.")

if __name__ == "__main__":
    asyncio.run(init_db())
