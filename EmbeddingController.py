from fastapi import FastAPI
from BO.QuestionInput import QuestionInput
from BO.SelectCategoryDataSet import SelectCategoryDataSet
from BO.SelectDataSet import SelectDataSet
from service.EmbeddingService import EmbeddingService






""" **************************************** Commande pour démarrer l'application **************************************** """

# uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8008
# uvicorn EmbeddingController:app --reload --workers 1 --host 0.0.0.0 --port 8011






""" **************************************** Commande pour installer les dépendances **************************************** """

"""
pip install -qU \
    transformers==4.30.2 \
    torch \
    einops==0.6.1 \
    accelerate==0.20.3 \
    datasets==2.14.5 \
    chromadb \
    sentence-transformers==2.2.2
"""






""" **************************************** Chargement de l'Api **************************************** """

app = FastAPI()
embedding_service = EmbeddingService()






""" **************************************** Api de test **************************************** """

@app.get("/ping")
async def pong():
    return {"ping": "pong!"}






""" **************************** Manipulation du modèle LLM **************************** """



@app.post("/load-dataset", response_model=bool, status_code=200)
async def load_dataset(data: SelectDataSet):
    """ Méthode qui initialise le dataset """
    if data.path is None:
        data.path = "resources/loaded_embedded_file/embedded_file.jsonl"
    embedded_dataset = embedding_service.dataset_init(data.path)
    return embedded_dataset



@app.post("/select-category", response_model=bool, status_code=200)
async def select_category(category_from_dataset: SelectCategoryDataSet):
    """ Méthode qui sélectionne des données du dataset en fonction de leur catégorie """
    if category_from_dataset.category is None:
        category_from_dataset.category = "closed_qa"
    embedded_dataset = embedding_service.select_data_category_from_dataset(category_from_dataset.category)
    return embedded_dataset



@app.post("/get-llm-embedding-answer", response_model=str, status_code=200)
async def get_llm_embedding_answer(input_question: QuestionInput):
    """ Méthode qui récupère une réponse à partir de la question posée """
    print(input_question.question)
    answer = embedding_service.get_llm_embedding_answer(input_question)
    return answer

