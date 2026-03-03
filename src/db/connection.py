from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.db.settings import settings

# Engine SQLAlchemy
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    echo=False
)

# Session factory (utile pour FastAPI plus tard)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_engine():
    return engine