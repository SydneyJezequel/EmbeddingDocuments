from pydantic import BaseModel






class SelectDataSet(BaseModel):
    """ Classe qui indique le chemin du dataset """

    path: str

