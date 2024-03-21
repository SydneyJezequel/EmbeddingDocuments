from pydantic import BaseModel






class SelectCategoryDataSet(BaseModel):
    """ Classe qui indique la 'Catégorie' de données à récupérer """

    category: str

