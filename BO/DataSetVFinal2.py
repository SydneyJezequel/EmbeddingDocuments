from datasets import load_dataset
import pandas as pd
import json






""" Classe qui représente le DataSet - VERSION FINALE.
Elle charge séparement :
- Le DataSet.
- Certains type de données pour contextualiser les réponses.
"""
class DataSetFinal2:



    """ Constructeur """
    def __init__(self):
        pass



    """ Méthode pour charger les données """
    """ ************* CETTE METHODE SERA SÛREMENT A SUPPRIMER ************* """
    def dataset_loader(self):
        # Chargement du dataset d'entrainement :
        train_dataset = load_dataset("databricks/databricks-dolly-15k", split='train')
        return train_dataset



    """ Méthode pour charger les données depuis un fichier """
    def dataset_loader_from_file(self, file_path):
        # Récupération de l'extension du fichier :
        file_extension = file_path.split('.')[-1].lower()
        # Chargement du dataset en fonction de l'extension :
        if file_extension == 'csv':
            # Chargement du fichier Csv avec pandas :
            dataset = pd.read_csv(file_path)
        elif file_extension == 'xlsx':
            # Chargement du fichier Excel avec pandas :
            dataset = pd.read_excel(file_path)
        elif file_extension == 'jsonl':
            # Chargement du fichier JSONL en lisant chaque ligne :
            with open(file_path, 'r') as file:
                lines = file.readlines()
            dataset = [json.loads(line.strip()) for line in lines]
        else:
            raise ValueError("Extension de fichier non prise en charge. Utiliser un fichier Csv, Excel (xlsx) ou Jsonl.")
        return dataset

