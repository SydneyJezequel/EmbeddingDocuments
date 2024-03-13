from monsterapi import client
from translate import Translator






# URL de l'Api : https://monsterapi.ai/user/playground?model=falcon-7b-instruct
""" Modèle LLM (Llama2-7B ou Falcon7B) """
class LlmModel:



    """ Constructeur """
    def __init__(self):
        # Définition du modèle :
        self.model_name = 'falcon-7b-instruct'
        # self.model_name = 'llama2-7b-chat'
        # Initialisation des pipeline, tokenizer et modèle :
        self.monster_client = self.initialize_model()
        # Sélection de la langue :
        self.prompt_language = 'français'
        self.translate_language = 'fr'



    """ Méthode qui initialise le modèle """
    def initialize_model(self):
        # Initialisation du client MonsterAPI avec la clé API :
        monster_api_key = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6ImM2OWRiZjIyNjMyYzE0ZjA2YThiNjEwZmQ2OGRiYzIzIiwiY3JlYXRlZF9hdCI6IjIwMjQtMDMtMTFUMjE6Mzc6MjguNTMzNTg5In0.kTwV0eh4EZs-ajLuUSPy1fTiSckXVn62xkmyZiw2H1Y'
        monster_client = client(monster_api_key)
        # Renvoi du pipeline, du tokenizer et du client MonsterAPI :
        return monster_client



    # Sélection de la langue :
    """ Méthode qui définit la langue des réponses renvoyées par le modèle """
    def set_language(self, language):
        self.prompt_language = language



    """ Méthode qui interroge le modèle Falcon7B """
    def generate_answer(self, question):
        # Interrogation du modèle :
        input_data = {
            'prompt': question,
            'language': self.prompt_language
        }
        print("input data : ", input_data)
        # Génération de la réponse :
        output = self.monster_client.generate(self.model_name, input_data)
        # Récupération de la réponse :
        response_text = output['text']
        print("log de réponse : ", response_text)
        return response_text



    """ Méthode qui interroge le modèle Falcon7B en y ajoutant un contexte """
    def generate_enriched_answer(self, question, context=None):
        # Préparation du prompt :
        prompt = question if context is None else f"{context}\n\n{question}"
        print("TEST PROMPT : ", prompt)
        # Génération des réponses avec le modèle Llama 2 via Replicate
        enriched_answer = self.generate_answer(prompt)
        answer_translated = self.translate_answer(enriched_answer)
        print("answer_translated : ", answer_translated)
        # Retour de la réponse :
        return answer_translated



    """ Méthode qui traduit la réponse du modèle en français. 
    Elle garantie que la réponse sera en langue française """
    def translate_answer(self, text):
        translator = Translator(to_lang=self.translate_language)
        translation = translator.translate(text)
        return translation