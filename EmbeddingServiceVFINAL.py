from BO.VectorStore import VectorStore
from BO.DataSet import DataSet
from BO.Llama2Model import Llama2Model






""" Classe qui représente la Base de données """
class EmbeddingService:



    """ Constructeur """
    def __init__(self, collection_name):
        pass



    """ Méthode qui initialise le dataSet """
    def dataset_init(self, file_path, category):
        dataset = DataSet()
        embedded_dataset = dataset.dataset_loader_from_file(file_path=file_path, category=category)
        return embedded_dataset



    """ Méthode qui initialise le VectorStore Chroma DB """
    def vector_store_init(self):
        vector_store = VectorStore("knowledge-base")
        return vector_store



    """ Méthode qui initialise le dataSet """
    def dataset_init(self, file_path, category):
        dataset = DataSet()
        embedded_dataset = dataset.dataset_loader_from_file(file_path=file_path, category=category)
        return embedded_dataset



    """ Méthode qui initialise le VectorStore Chroma DB """
    def vector_store_init(self):
        vector_store = VectorStore("knowledge-base")
        return vector_store



    """ Méthode qui initialise le Modèle LLM """
    def llm_model_init(self, token):
        token = "r8_JDzPiCeExTDt9TR6t5wkXoFoGLLerJ63V0bCG"
        llmModel = Llama2Model(token)
        return llmModel



    """ Chargement du dataset dans le VectorStore Chroma DB """
    def load_dataset_into_vector_store(self, vector_store, file_path, category):
        try:
            embedded_dataset = self.dataset_init(self, file_path, category)
            vector_store.populate_vectors(embedded_dataset)
            print("dataset chargé dans le vector store. ")
            return True
        except Exception as e:
            print(f"Une erreur s'est produite : {e}")
            return False



    """ Méthode qui répond aux questions """
    def get_answer(self, question, llmModel, vector_store):
        # Récupération du context dans le VectorStore :
        context_response = vector_store.search_context(question)
        # Extraction du contexte de la réponse :
        context= "".join(context_response['documents'][0])
        # Génération de la réponse :
        response= llmModel.generate_enriched_answer(question, context=context)
        print("Réponse Camélia : ")
        return response

