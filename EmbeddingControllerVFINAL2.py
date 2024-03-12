from fastapi import FastAPI
from BO.QuestionInput import QuestionInput
from BO.SelectCategoryDataSetVFinal2 import SelectCategoryDataSetVFinal2
from BO.SelectDataSetVFinal2 import SelectDataSetVFinal2
from EmbeddingServiceVFINAL2 import EmbeddingServiceVFINAL2






""" **************************************** Commande pour démarrer l'application **************************************** """

# uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8008
# uvicorn EmbeddingControllerVFINAL2:app --reload --workers 1 --host 0.0.0.0 --port 8011






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
embedding_service = EmbeddingServiceVFINAL2()






""" **************************************** Api de test **************************************** """

@app.get("/ping")
async def pong():
    return {"ping": "pong!"}






""" **************************** Manipulation du modèle LLM **************************** """



""" Méthode qui initialise le dataset """
@app.post("/load_dataset", response_model=bool, status_code=200)
async def load_dataset(data: SelectDataSetVFinal2):
    if data.file_path is None:
        data.file_path = "./embedded_file/camelia_yvon_jezequel_dataset.jsonl"
    embedded_dataset = embedding_service.dataset_init(data.file_path)
    return embedded_dataset



""" Méthode qui charge une catégorie de données du dataset """
@app.get("/select_category", response_model=bool, status_code=200)
async def select_category(category_from_dataset: SelectCategoryDataSetVFinal2):
    if category_from_dataset.category is None:
        category_from_dataset.category = "closed_qa"
    embedded_dataset = embedding_service.select_data_category_from_dataset(category_from_dataset.category)
    return embedded_dataset



""" Méthode qui récupère une réponse """
@app.get("/get_llm_embedding_answer", response_model=str, status_code=200)
async def get_llm_embedding_answer(input_question: QuestionInput):
    answer = embedding_service.get_llm_embedding_answer(input_question.question)
    return answer

