from BO.VectorStore import VectorStore
from BO.DataSet import DataSet
from BO.Llama2Model import Llama2Model










""" ***************************************** Test sur un dataset extérieur en jsonl ***************************************** """


# 1- Création du dataset :
dataset = DataSet()
file_path = "./embedded_file/camelia_yvon_jezequel_dataset.jsonl"
camelia_yvon_jezequel_dataset = dataset.dataset_loader_from_file(file_path=file_path)
print("dataset : ")
print(camelia_yvon_jezequel_dataset)


# 2- Initialisation du Vector Store :
vector_store = VectorStore("knowledge-base")


# 3- Stockage du dataSet d'entrainement dans le Vector Store :
vector_store.populate_vectors(camelia_yvon_jezequel_dataset)


# 4- Questions :
question_camelia = "Qui a créé le Camélia Yvon Jézéquel ? "
question_obtention = "Qui a fait reconnaitre le Camélia Yvon Jézéquel ? "


# 5- Récupération du context dans le VectorStore :
context_response = vector_store.search_context(question_camelia)
print(context_response)


# 6- Extraction du contexte de la réponse :
context_camelia = "".join(context_response['documents'][0])


# 7- Génération d'une réponse avec son contexte :
token = "r8_JDzPiCeExTDt9TR6t5wkXoFoGLLerJ63V0bCG"
llama2ModelCamelia = Llama2Model(token)
reponse_camelia = llama2ModelCamelia.generate_enriched_answer(question_camelia, context=context_camelia)
print("Réponse Camélia : ")
print(reponse_camelia)
reponse_obtention = llama2ModelCamelia.generate_enriched_answer(question_obtention, context=context_camelia)
print("Réponse Obtention : ")
print(reponse_camelia)
