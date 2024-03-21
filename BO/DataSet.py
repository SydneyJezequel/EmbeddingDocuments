import pandas as pd
import json






class DataSet:
    """ Classe qui représente le DataSet """
    """
        Elle charge séparement :
        * Le Dataset.
        * Certaines catégories de données pour contextualiser les réponses.
    """



    def __init__(self):
        """ Constructeur """
        pass



    def dataset_loader_from_file(self, file_path):
        """ Méthode pour charger les données depuis un fichier """
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

