from pydantic import BaseModel






""" Classe qui représente une question posée """
class QuestionInput(BaseModel):
    question: str
    category: str

