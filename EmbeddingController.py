from fastapi import FastAPI






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

