import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from monsterapi import client






# URL : https://monsterapi.ai/user/playground?model=falcon-7b-instruct
""" Modèle Falcon7B """
class Falcon7BModel:



    """ Constructeur """
    def __init__(self):
        # Définition du modèle :
        self.model_name = 'falcon-7b-instruct'
        # Initialisation des pipeline, tokenizer et modèle :
        self.monster_client = self.initialize_model()



    """ Méthode qui initialise le modèle """
    def initialize_model(self):
        # Initialisation du client MonsterAPI avec la clé API :
        monster_api_key = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6ImM2OWRiZjIyNjMyYzE0ZjA2YThiNjEwZmQ2OGRiYzIzIiwiY3JlYXRlZF9hdCI6IjIwMjQtMDMtMTFUMjE6Mzc6MjguNTMzNTg5In0.kTwV0eh4EZs-ajLuUSPy1fTiSckXVn62xkmyZiw2H1Y'
        monster_client = client(monster_api_key)
        # Renvoi du pipeline, du tokenizer et du client MonsterAPI :
        return monster_client



    """ Méthode qui interroge le modèle Falcon7B """
    def generate_answer(self, question):
        # Attributs :
        result = ""
        # Traitement :
        # Interrogation du modèle :
        input_data = {
            'prompt': question
        }
        output = self.monster_client.generate(self.model_name, input_data)
        print("TEST INPUT_DATA : ", input_data)
        print("TEST OUTPUT : ", output)
        response_text = output['text']
        print("TEST OUTPUT 2 : ", response_text)
        # Récupération de la réponse :
        for item in output:
            result += item
        return response_text



    """ Méthode qui interroge le modèle Falcon7B en y ajoutant un contexte """
    def generate_enriched_answer(self, question, context=None):
        # Préparation du prompt :
        prompt = question if context is None else f"{context}\n\n{question}"
        print("TEST PROMPT : ", prompt)
        # Génération des réponses avec le modèle Llama 2 via Replicate
        enriched_answer = self.generate_answer(prompt)
        # Retour de la réponse :
        return enriched_answer

