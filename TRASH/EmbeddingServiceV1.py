from BO.VectorDataBase import VectorStore
from TRASH.TRASH_SERVICE_CONTROLLER.DataSet import DataSet
import os
import replicate






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






"""  *******************  Configuration de l'API ******************* """


# Méthode qui configure l'Api :
def api_config(token):
    os.environ["REPLICATE_API_TOKEN"] = token
    api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    return api



# Méthode qui interroge l'Api du modèle Llama 2 :
def get_answer(question, api):

    # Attributs :
    result = ""

    # Traitements :
    output = api.run(
        "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
        input={"prompt": question}
    )
    # Affichage et chargement du résultat :
    print(f"Result: {output}")
    for item in output:
        print(item, end="")
        result = item
    # Retour de la réponse :
    return result



# Méthode qui interroge l'Api du modèle Llama 2 :
def get_enriched_answer(question, context, api):

    # Attributs :
    result = ""

    # Traitements :
    output = api.run(
        "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
        input={"prompt": question, "context": context}
    )
    # Affichage et chargement du résultat :
    print(f"Result: {output}")
    for item in output:
        print(item, end="")
        result = item
    # Retour de la réponse :
    return result






""" ***************** Exécution du traitement ***************** """



# Token de l'Api :
token = "r8_JDzPiCeExTDt9TR6t5wkXoFoGLLerJ63V0bCG"

# Configuration de l'Api :
api = api_config(token)

# TEST - Génération d'une réponse :
question = "Qui est le personnage principal dans star wars ?"
print("QUESTION : ", question)
answer = get_answer(question, api)
print('REPONSE : ', answer)






""" ***************** Génération de Réponses intégrant le contexte ***************** """


# Cette étape a lieue après l'initialisation des vector_store et falcon_model.

# Question :
question = "When was Tomoaki Komorida born?"

# Récupération du context dans le VectorStore :
context_response = vector_store.search_context(question)

# Extraction du contexte de la réposnse. C'est le 1er élément du 'context' key :
context = "".join(context_response['context'][0])

# Génération d'une réponse en utilisant le modèle FalconB et le contexte :

enriched_answer = get_enriched_answer(question, context, api)
print(f"Result: {enriched_answer}")






