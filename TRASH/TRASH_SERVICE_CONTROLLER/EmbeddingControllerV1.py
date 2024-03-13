from fastapi import FastAPI
from BO.QuestionInput import QuestionInput
from TRASH.TRASH_SERVICE_CONTROLLER.SelectDataSet import SelectDataSet
from EmbeddingServiceVFINAL import EmbeddingService






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



""" Méthode qui initialise un dataset """
@app.post("/load_dataset", response_model=bool, status_code=200)
async def load_dataset(data: SelectDataSet):
    if data.file_path is None:
        data.file_path = "../../embedded_file/camelia_yvon_jezequel_dataset.jsonl"
    if data.category is None:
        data.category = "closed_qa"
    embedded_dataset = embedding_service.load_dataset_into_vector_store(data.file_path, data.category)
    return embedded_dataset



""" Méthode qui sélectionne un modèle """
@app.get("/select_llm", response_model=bool, status_code=200)
async def select_llm(token: str = "r8_JDzPiCeExTDt9TR6t5wkXoFoGLLerJ63V0bCG"):
    llm_model = embedding_service.llm_model_init(token)
    return llm_model



""" Méthode qui récupère une réponse """
@app.get("/get_llm_embedding_answer", response_model=str, status_code=200)
async def get_llm_embedding_answer(input_question: QuestionInput):
    answer = embedding_service.get_llm_embedding_answer(input_question.question)
    return answer

