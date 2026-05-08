# Import the Base class
from app.db.base_class import Base # noqa

# Import all models here so that they are registered on the metadata
from app.db.models.user_model import User # noqa
from app.db.models.question_model import Question # noqa
from app.db.models.test_model import VocationalTestResult # noqa
