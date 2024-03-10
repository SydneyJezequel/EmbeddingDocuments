from BO.VectorStore import VectorStore
from BO.DataSet import DataSet
from BO.Llama2Model import Llama2Model






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






"""


# Classe Embedding Service : VERSION avec les fonctionnalités du commit 8 :
   Commit 8 : Mise en place du controller et des Api pour manipuler l'Embedding.
    Fonctionnalités :
    -Charger le modèle
    -Sélectionner un type de données du modèle
    -Générer une réponse



class EmbeddingService:



    # Constructeur 
    def __init__(self):
        self.dataset = DataSet()
        self.embedded_dataset = DataSet()
        self.vector_store = VectorStore("knowledge-base")
        self.llm_model = Llama2Model("r8_JDzPiCeExTDt9TR6t5wkXoFoGLLerJ63V0bCG")



    #  Méthode qui initialise le dataSet 
    def dataset_init(self, file_path):
        dataset = DataSet()
        self.dataset = dataset.dataset_loader_from_file(file_path=file_path)
        print("dataset_init - embedded dataset : ")
        print(self.dataset)
        return self.dataset



    #  Méthode qui renvoie un dataset qui contient une catégorie de données 
    def select_category(self, category):
        try:
            # Filtre sur les entrées de type 'closed_qa' :
            self.embedded_dataset = self.dataset.filter(lambda example: example['category'] == category)
        except Exception as e:
            print(f"Une erreur s'est produite : {e}")
            return False

"""


""" Classe DataSet - Méthode pour charger les données depuis un fichier """
"""

    def dataset_loader_from_file(self, file_path, category='closed_qa'):
        # Vérifie l'extension du fichier pour déterminer le type :
        file_extension = file_path.split('.')[-1].lower()
        # Chargement du dataset en fonction du type de fichier :
        if file_extension == 'csv':
            # Charger le fichier Csv avec pandas :
            dataset = pd.read_csv(file_path)
        elif file_extension == 'xlsx':
            # Charger le fichier Excel avec pandas :
            dataset = pd.read_excel(file_path)
        elif file_extension == 'jsonl':
            # Charger le fichier JSONL en lisant chaque ligne :
            with open(file_path, 'r') as file:
                lines = file.readlines()
            dataset = [json.loads(line.strip()) for line in lines]
        else:
            raise ValueError("Extension de fichier non prise en charge. Utiliser un fichier CSV, Excel (xlsx) ou JSONL.")
        return dataset

"""