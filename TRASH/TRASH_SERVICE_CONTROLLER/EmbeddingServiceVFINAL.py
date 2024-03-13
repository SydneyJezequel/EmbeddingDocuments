from BO.VectorDataBase import VectorStore
from TRASH.TRASH_SERVICE_CONTROLLER.DataSet import DataSet
from TRASH.TRASH_SERVICE_CONTROLLER.Llama2Model import Llama2Model






""" Classe qui représente la Base de données """
class EmbeddingService:



    """ Constructeur """
    def __init__(self):
        self.embedded_dataset = DataSet()
        self.vector_store = VectorStore("knowledge-base")
        self.llm_model = Llama2Model("r8_JDzPiCeExTDt9TR6t5wkXoFoGLLerJ63V0bCG")



    """ Méthode qui initialise le dataSet """
    def dataset_init(self, file_path, category):
        dataset = DataSet()
        self.embedded_dataset = dataset.dataset_loader_from_file(file_path=file_path, category=category)
        print("EMBEDDED_DATASET : ")
        print(self.embedded_dataset)
        return self.embedded_dataset



    """ Méthode qui initialise le Modèle LLM """
    def llm_model_init(self, token):
        """
        METHODE A IMPLEMENTER AVEC LE TOKEN & L'URL POUR PERMETTRE A L'UTILISATEUR
        DE CHOISIR LE LLM UTILISE.
        """
        try:
            token = "r8_JDzPiCeExTDt9TR6t5wkXoFoGLLerJ63V0bCG"
            llmModel = Llama2Model(token)
            return True
        except Exception as e:
            print(f"Une erreur s'est produite : {e}")
            return False



    """ Chargement du dataset dans le VectorStore Chroma DB """
    def load_dataset_into_vector_store(self, file_path, category):
        try:
            self.embedded_dataset = self.dataset_init(file_path, category)
            print("EMBEDDED_DATASET - VECTOR STORE : ")
            print(self.embedded_dataset)
            self.vector_store.populate_vectors(self.embedded_dataset)
            print("dataset chargé dans le vector store. ")
            return True
        except Exception as e:
            print(f"Une erreur s'est produite : {e}")
            return False



    """ Méthode qui répond aux questions """
    def get_llm_embedding_answer(self, question):
        # Récupération du context dans le VectorStore :
        context_response = self.vector_store.search_context(question)
        print("1")
        # Extraction du contexte de la réponse :
        context = "".join(context_response['documents'][0])
        print("2")
        # Génération de la réponse :
        response = self.llm_model.generate_enriched_answer(question, context=context)
        print("3")
        print("Réponse  : ")
        print(response)
        return response

