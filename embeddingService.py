from datasets import load_dataset
from VectorStore import VectorStore
from DataSet import DataSet






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






""" **************************** Exécution du Service **************************** """

# Initialisation du DataSet :
dataset = DataSet()

# Chargement du dataset d'entrainement :
closed_qa_dataset = dataset.dataset_loader()

# Test :
print(closed_qa_dataset[0])

# Initialisation du Vector Store :
vector_store = VectorStore("knowledge-base")

# Stockage du dataSet d'entrainement dans le Vector Store :
vector_store.populate_vectors(closed_qa_dataset)

# Test :
print(vector_store)
