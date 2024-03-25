import chromadb
from sentence_transformers import SentenceTransformer

import config


class VectorDataBase:
    """ Classe qui représente la Base de données """



    def __init__(self, collection_name):
        """ Constructeur """
        # Initialisation du modèle d'Embedding :
        self.embedding_model = SentenceTransformer(config.TRANSFORMER)
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name=collection_name)



    def populate_vectors(self, dataset):
        """ Méthode pour enregistrer des vecteurs en BDD """
        for i, item in enumerate(dataset):
            combined_text = f"{item['instruction']}. {item['context']}"
            embeddings = self.embedding_model.encode(combined_text).tolist()
            self.collection.add(embeddings=[embeddings], documents=[item['context']], ids=[f"id_{i}"])



    def search_context(self, query, n_results=1):
        """ Méthode pour rechercher des vecteurs à partir d'une requête """
        query_embeddings = self.embedding_model.encode(query).tolist()
        return self.collection.query(query_embeddings=query_embeddings, n_results=n_results)

