from pydantic import BaseModel






""" Version final - Classe qui indique le chemin du dataset  """
class SelectDataSet(BaseModel):
    path: str

