from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.base import Base

class VocationalTestResult(Base):
    __tablename__ = "vocational_test_results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    user = relationship("User", backref="test_results")
    
    # Store the scores (RIASEC) as JSON
    scores = Column(JSON, nullable=False)
    
    # The recommended career category
    recommendation = Column(String, nullable=False)
    
    # Additional details or justification
    details = Column(String)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
