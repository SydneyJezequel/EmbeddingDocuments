from datasets import load_dataset






""" Classe qui représente la Base de données """
class DataSet:



    """ Constructeur """
    def __init__(self):
        pass



    """ Méthode pour charger les données """
    def dataset_loader(self):
        # Chargement du dataset d'entrainement :
        train_dataset = load_dataset("databricks/databricks-dolly-15k", split='train')
        # Filtre sur les entrées de type 'closed_qa' :
        closed_qa_dataset = train_dataset.filter(lambda example: example['category'] == 'closed_qa')
        return closed_qa_dataset

