from pydantic import BaseModel






""" Classe qui indique le chemin du dataset et le type de donées à récupérée dedans """
class SelectDataSet(BaseModel):
    file_path: str
    category: str