from BO.DataSetVFinal2 import DataSetFinal2
from BO.LlmModel import LlmModel
from BO.VectorDataBase import VectorDataBase






""" Classe EmbeddingService - VERSION FINALE """
class EmbeddingServiceVFINAL2:



    """ Constructeur """
    def __init__(self):
        # Initialisation du DataSet brut :
        self.dataset = []
        # Initialisation du DataSet filtré :
        self.embedded_dataset = DataSetFinal2()
        # Initialisation du modèle d'Embedding :
        self.vector_store = VectorDataBase("knowledge-base")
        # Initialisation du modèle LLM :
        # self.llm_model = Llama2Model("r8_JDzPiCeExTDt9TR6t5wkXoFoGLLerJ63V0bCG")
        self.llm_model = LlmModel()



    """ Méthode qui initialise le dataset """
    def dataset_init(self, file_path):
        try:
            dataset = DataSetFinal2()
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
            # ***************** TEST ***************** #
            print("select_data_category_from_dataset: ", self.embedded_dataset)
            # ***************** TEST ***************** #
            self.load_dataset_into_vector_store()
            return True
        except Exception as e:
            print(f"Une erreur s'est produite : {e}")
            return False



    """ Chargement du dataset dans Chroma DB """
    def load_dataset_into_vector_store(self):
        try:
            # ***************** TEST ***************** #
            print("load_dataset_into_vector_store : ", self.embedded_dataset)
            # ***************** TEST ***************** #
            self.vector_store.populate_vectors(self.embedded_dataset)
            print("dataset chargé dans le vector store. ")
            return True
        except Exception as e:
            print(f"Une erreur s'est produite : {e}")
            return False



    """ Méthode qui répond aux questions """
    def get_llm_embedding_answer(self, question):
        # Récupération du context dans la BDD Vectorielle :
        context_response = self.vector_store.search_context(question)
        # Extraction du contexte de la réponse :
        context = "".join(context_response['documents'][0])
        # Génération de la réponse :
        response = self.llm_model.generate_enriched_answer(question, context=context)
        print("Réponse  : ", response)
        return response

