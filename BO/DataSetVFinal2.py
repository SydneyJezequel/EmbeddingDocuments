from datasets import load_dataset
import pandas as pd
import json






""" Classe qui représente la Base de données - Versional Final.
Elle charge séparement :
- Le DataSet.
- Certains type de données pour contextualiser les réponses.
"""
class DataSetFinal2:



    """ Constructeur """
    def __init__(self):
        pass



    """ Méthode pour charger les données """
    def dataset_loader(self):
        # Chargement du dataset d'entrainement :
        train_dataset = load_dataset("databricks/databricks-dolly-15k", split='train')
        return train_dataset



    """ Méthode pour charger les données depuis un fichier """
    def dataset_loader_from_file(self, file_path):
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

