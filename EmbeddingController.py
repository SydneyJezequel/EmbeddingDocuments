from fastapi import FastAPI
from EmbeddingServiceVFINAL import EmbeddingService






""" **************************************** Commande pour démarrer l'application **************************************** """

# uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8008
# uvicorn EmbeddingController:app --reload --workers 1 --host 0.0.0.0 --port 8011






""" **************************************** Chargement de l'Api **************************************** """

app = FastAPI()






""" **************************************** Api de test **************************************** """

@app.get("/ping")
async def pong():
    return {"ping": "pong!"}






""" **************************************** Exécution du modèle GAN **************************************** """

@app.get("/ping")
async def load_dataset():
    file_path = ""
    category =""
    embedding_service = EmbeddingService()
    vector_store = embedding_service.vector_store_init()
    embedding_service.dataset_init(file_path, category)
    embedding_service.load_dataset_into_vector_store(vector_store, file_path, category)
    return {"ping": "pong!"}



@app.get("/ping")
async def select_llm():
    token = ""
    llm_selected = ""
    embedding_service = EmbeddingService()
    llm_model = embedding_service.llm_model_init(token)
    return {"ping": "pong!"}



@app.get("/ping")
async def get_answer():
    question = ""
    llModel = ""
    vector_store = ""
    embedding_service = EmbeddingService()
    embedding_service.get_answer(question, llmModel, vector_store)
    return {"ping": "pong!"}






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

