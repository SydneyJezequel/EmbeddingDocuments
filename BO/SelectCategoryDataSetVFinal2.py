from pydantic import BaseModel






"""  Version final - Classe qui indique la 'Catégorie' de données à récupérer """
class SelectCategoryDataSetVFinal2(BaseModel):
    category: str

