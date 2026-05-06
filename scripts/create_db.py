import asyncio
import asyncpg
import os

async def create_db():
    user = "postgres"
    password = "admin"
    host = "localhost"
    database = "vocational_db"

    print(f"Connecting to postgres default database...")
    try:
        conn = await asyncpg.connect(user=user, password=password, host=host, database="postgres")
        
        # Check if database exists
        exists = await conn.fetchval(f"SELECT 1 FROM pg_database WHERE datname = '{database}'")
        
        if not exists:
            print(f"Creating database {database}...")
            # We cannot create database inside a transaction, so we use the connection directly
            await conn.execute(f'CREATE DATABASE {database}')
            print(f"Database {database} created.")
        else:
            print(f"Database {database} already exists.")
            
        await conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(create_db())
