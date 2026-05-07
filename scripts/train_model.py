import asyncio
from app.services.ml_service import ml_service

async def run_training():
    print("Starting training process via MLService...")
    stats = await ml_service.train_from_files()
    print(f"Training complete! Accuracy: {stats['accuracy']:.4f}")
    print(f"Model saved to app/ml/assets/")

if __name__ == "__main__":
    asyncio.run(run_training())
