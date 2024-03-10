from fastapi import FastAPI
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






""" **************************************** Exécution du modèle GAN **************************************** """



@app.post("/load_dataset", response_model=bool, status_code=200)
async def load_dataset(file_path: str = "./embedded_file/camelia_yvon_jezequel_dataset.jsonl", category: str = "closed_qa"):
    embedded_dataset = embedding_service.load_dataset_into_vector_store(file_path, category)
    return embedded_dataset



@app.get("/select_llm", response_model=bool, status_code=200)
async def select_llm(token: str = "r8_JDzPiCeExTDt9TR6t5wkXoFoGLLerJ63V0bCG"):
    llm_model = embedding_service.llm_model_init(token)
    return llm_model



@app.get("/get_llm_embedding_answer", response_model=str, status_code=200)
async def get_answer(question=str):
    answer = embedding_service.get_answer(question)
    return answer

