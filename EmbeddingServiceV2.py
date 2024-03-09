from BO.VectorStore import VectorStore
from BO.DataSet import DataSet
from BO.Llama2Model import Llama2Model










""" ***************************************** Test sur un dataset extérieur en jsonl ***************************************** """






# 1- Création du dataset :
dataset = DataSet()
file_path = "./embedded_file/camelia_yvon_jezequel_dataset.jsonl"
category = 'closed_qa'
camelia_yvon_jezequel_dataset = dataset.dataset_loader_from_file(file_path=file_path, category=category)









# 2- Initialisation du Vector Store :
vector_store = VectorStore("knowledge-base")






# 3- Stockage du dataSet d'entrainement dans le Vector Store :
vector_store.populate_vectors(camelia_yvon_jezequel_dataset)






# 4- Initialisation du modèle :
token = "r8_JDzPiCeExTDt9TR6t5wkXoFoGLLerJ63V0bCG"
llama2ModelCamelia = Llama2Model(token)






# 5- Questions / Répones :



# QUESTION 1 :

# Question création du Camélia :
question_camelia = "Qui a créé le Camélia Yvon Jézéquel ? "
print(question_camelia)

# Récupération du context dans le VectorStore :
context_response_camelia = vector_store.search_context(question_camelia)

# Extraction du contexte de la réponse :
context_camelia = "".join(context_response_camelia['documents'][0])

# Génération de la réponse :
reponse_camelia = llama2ModelCamelia.generate_enriched_answer(question_camelia, context=context_camelia)
print("Réponse Camélia : ")
print(reponse_camelia)


# QUESTION 2 :

# Question reconaissance du Camélia :
question_obtention = "Qui a fait reconnaitre le Camélia Yvon Jézéquel ? "
print(question_obtention)

# Récupération du context dans le VectorStore :
context_response_obtention = vector_store.search_context(question_obtention)

# Extraction du contexte de la réponse :
context_obtention = "".join(context_response_obtention['documents'][0])

# Génération de la réponse avec son contexte :
reponse_obtention = llama2ModelCamelia.generate_enriched_answer(question_obtention, context=context_obtention)
print("Réponse Obtention : ")
print(reponse_obtention)




# QUESTION 3 :

# Modification du Dataset :
file_path = "./embedded_file/camelia_yvon_jezequel_dataset.jsonl"
category = 'brainstorming'
camelia_yvon_jezequel_poem_dataset = dataset.dataset_loader_from_file(file_path=file_path, category=category)

# Stockage du dataSet d'entrainement dans le Vector Store :
vector_store.populate_vectors(camelia_yvon_jezequel_poem_dataset)


# Question création du Camélia :
question_poem_camelia = "Rédige moi un poème au sujet du Camélia Yvon Jézéquel ? "
print(question_poem_camelia)

# Récupération du context dans le VectorStore :
context_response_poem_camelia = vector_store.search_context(question_poem_camelia)

# Extraction du contexte de la réponse :
context_poem_camelia = "".join(context_response_poem_camelia['documents'][0])

# Génération de la réponse :
reponse_poem_camelia = llama2ModelCamelia.generate_enriched_answer(question_poem_camelia, context=context_poem_camelia)
print("Poème sur le Camélia Yvon Jézéquel : ")
print(reponse_poem_camelia)



