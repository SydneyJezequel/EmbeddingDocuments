from pydantic import BaseModel






class QuestionInput(BaseModel):
    """ Classe qui représente une question posée """

    question: str

