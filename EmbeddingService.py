from BO.VectorStore import VectorStore
from BO.DataSet import DataSet
from BO.Falcon7BModel import Falcon7BModel






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






""" **************************** Base de connnaissance **************************** """


"""
# Chargement du dataset d'entrainement :
train_dataset = load_dataset("databricks/databricks-dolly-15k", split='train')

# Filtre sur les entrées de type 'closed_qa' :
closed_qa_dataset = train_dataset.filter(lambda example: example['category'] == 'closed_qa')

print(closed_qa_dataset[0])
"""






""" **************************** Utilisation de la Classe Vector Store **************************** """


"""
if __name__ == "__main__":
    # Initialisation du Vector Store :
    vector_store = VectorStore("knowledge-base")
    # Stockage du dataSet d'entrainement dans le Vector Store :
    vector_store.populate_vectors(closed_qa_dataset)
"""





""" ****************************************************************************** """
""" **************************** Exécution du Service **************************** """
""" ****************************************************************************** """






""" **************************** Initialisation du DataSet **************************** """


# Initialisation du DataSet :
dataset = DataSet()

# Chargement du dataset d'entrainement :
closed_qa_dataset = dataset.dataset_loader()

# Test :
print(closed_qa_dataset[0])






""" **************************** Initialisation du Vector Store **************************** """


# Initialisation du Vector Store :
vector_store = VectorStore("knowledge-base")

# Stockage du dataSet d'entrainement dans le Vector Store :
vector_store.populate_vectors(closed_qa_dataset)

# Test :
# print(vector_store)






""" **************************** Utilisation du modèle Falcon7B **************************** """


# Initialisation du modèle Falcon7B
falcon_model = Falcon7BModel()

# Question :
user_question = "When was Tomoaki Komorida born?"

# Génération d'une question en utilisant le LLM :
answer = falcon_model.generate_answer(user_question)
print(f"Result: {answer}")






""" ***************** Génération de Réponses intégrant le contexte ***************** """


# Cette étape a lieue après l'initialisation des vector_store et falcon_model.

# Récupération du context dans le VectorStore :
context_response = vector_store.search_context(user_question)

# Extraction du contexte de la réposnse. C'est le 1er élément du 'context' key :
context = "".join(context_response['context'][0])

# Génération d'une réponse en utilisant le modèle FalconB et le contexte :
enriched_answer = falcon_model.generate_answer(user_question, context=context)
print(f"Result: {enriched_answer}")

