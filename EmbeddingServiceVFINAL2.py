from BO.DataSetVFinal2 import DataSetFinal2
from BO.Llama2Model import Llama2Model
from BO.VectorStore import VectorStore






""" Classe EmbeddingService avec la possibilité de """
class EmbeddingServiceVFINAL2:



    """ Constructeur """
    def __init__(self):
        self.dataset = []
        self.embedded_dataset = DataSetFinal2()
        self.vector_store = VectorStore("knowledge-base")
        self.llm_model = Llama2Model("r8_JDzPiCeExTDt9TR6t5wkXoFoGLLerJ63V0bCG")



    """ Méthode qui initialise le dataSet """
    def dataset_init(self, file_path):
        dataset = DataSetFinal2()
        self.dataset = dataset.dataset_loader_from_file(file_path=file_path)
        print("dataset_init - embedded dataset : ")
        print(self.dataset)
        return self.dataset



    """ Méthode charge un dataset contenant une catégorie de données """
    def select_data_category_from_dataset(self, category):
        try:
            filtered_examples = []
            for example in self.dataset:
                if example.get('category') == category and example.get('category') is not None:
                    filtered_examples.append(example)
            self.embedded_dataset = filtered_examples
            self.load_dataset_into_vector_store()
        except Exception as e:
            print(f"Une erreur s'est produite : {e}")
            return False



    """ Chargement du dataset dans le VectorStore Chroma DB """
    def load_dataset_into_vector_store(self):
        try:
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

