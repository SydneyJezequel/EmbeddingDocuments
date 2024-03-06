import chromadb
from sentence_transformers import SentenceTransformer






""" Classe qui représente la Base de données """
class VectorStore:



    """ Constructeur """
    def __init__(self, collection_name):
        # Initialisation du modèle d'Embedding :
        self.embedding_model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name=collection_name)



    """ Méthode pour enregistrer des vecteurs en BDD """
    def populate_vectors(self, dataset):
        for i, item in enumerate(dataset):
            combined_text = f"{item['instruction']}. {item['context']}"
            embeddings = self.embedding_model.encode(combined_text).tolist()
            self.collection.add(embeddings=[embeddings], documents=[item['context']], ids=[f"id_{i}"])



    """ Méthode pour rechercher des vecteurs à partir d'une requête """
    def search_context(self, query, n_results=1):
        query_embeddings = self.embedding_model.encode(query).tolist()
        return self.collection.query(query_embeddings=query_embeddings, n_results=n_results)

