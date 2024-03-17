from BO.DataSet import DataSet
from BO.LlmModel import LlmModel
from BO.VectorDataBase import VectorDataBase






""" Classe EmbeddingService - VERSION FINALE """
class EmbeddingService:



    """ Constructeur """
    def __init__(self):
        # Initialisation du DataSet brut :
        self.dataset = []
        # Initialisation du DataSet filtré :
        self.embedded_dataset = DataSet()
        # Initialisation du modèle d'Embedding :
        self.vector_store = VectorDataBase("knowledge-base")
        # Initialisation du modèle LLM :
        # self.llm_model = Llama2Model("r8_JDzPiCeExTDt9TR6t5wkXoFoGLLerJ63V0bCG")
        self.llm_model = LlmModel()



    """ Méthode qui initialise le dataset """
    def dataset_init(self, file_path):
        try:
            dataset = DataSet()
            self.dataset = dataset.dataset_loader_from_file(file_path=file_path)
            # ***************** TEST ***************** #
            print("dataset_init - embedded dataset : ", self.dataset)
            # ***************** TEST ***************** #
            return True
        except Exception as e:
            print(f"Une erreur s'est produite : {e}")
            return False



    """ Méthode qui charge une catégorie de données du dataset """
    def select_data_category_from_dataset(self, category):
        try:
            filtered_examples = []
            for example in self.dataset:
                if example.get('category') == category and example.get('category') is not None:
                    filtered_examples.append(example)
            self.embedded_dataset = filtered_examples
            print("EMBEDDED_DATASET : ", self.embedded_dataset)
            self.load_dataset_into_vector_store()
            return True
        except Exception as e:
            print(f"Une erreur s'est produite : {e}")
            return False



    """ Chargement du dataset dans Chroma DB """
    def load_dataset_into_vector_store(self):
        try:
            self.vector_store.populate_vectors(self.embedded_dataset)
            print("dataset chargé dans le vector store. ")
            return True
        except Exception as e:
            print(f"Une erreur s'est produite : {e}")
            return False



    """ Méthode qui répond aux questions """
    def get_llm_embedding_answer(self, input_question):
        # Préparation des paramètres de la question :
        question = input_question.question
        category = input_question.category
        # Récupération du context dans la BDD Vectorielle :
        context_response = self.vector_store.search_context(question)
        # Extraction du contexte de la réponse :
        context = "".join(context_response['documents'][0])
        print("CONTEXT : ", context)
        # Génération de la réponse :
        response = self.llm_model.generate_enriched_answer(question, category, context=context)
        return response

