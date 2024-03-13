from pydantic import BaseModel






"""  Version final - Classe qui indique la 'Catégorie' de données à récupérer """
class SelectCategoryDataSet(BaseModel):
    category: str

