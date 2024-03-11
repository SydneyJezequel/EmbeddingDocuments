from pydantic import BaseModel






""" Classe qui représente une question posée """
class SelectDataSet(BaseModel):
    file_path: str
    category: str