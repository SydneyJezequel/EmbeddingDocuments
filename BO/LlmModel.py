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
        }
        print("input data : ", input_data)
        # Génération de la réponse :
        output = self.monster_client.generate(self.model_name, input_data)
        # Récupération de la réponse :
        response_text = output['text']
        return response_text



    """ Méthode qui interroge le modèle Falcon7B en y ajoutant un contexte """
    def generate_enriched_answer(self, question, category, context=None):
        # Préparation du prompt :
        prompt = f'"question": "{question}", "context": "{context}", "language": "français"' if context is not None else f'"question": "{question}", "context": "", "language": "français"'
        # Initialisation des valeurs (category 'closed_qa' par défaut) :
        top_k = 15
        top_p = 0.1
        temp = 0.1
        max_length = 256
        beam_size = 1
        # Modification des paramètres passés à l'Api en fonction de la catégorie des données :
        print("category : ", category)
        if category == 'classification':
            top_k = 15
            top_p = 0.5
            temp = 0.99
            max_length = 256
            beam_size = 1
        if category == 'summarization':
            top_k = 15
            top_p = 0.5
            temp = 0.99
            max_length = 256
            beam_size = 1
        if category == 'brainstorming':
            top_k = 15
            top_p = 0.5
            temp = 0.99
            max_length = 256
            beam_size = 1
        # Préparation de l'objet envoyé à l'Api :
        input_data = {
            'prompt': prompt,
            'top_k': top_k,
            'top_p': top_p,
            'temp': temp,
            'max_length': max_length,
            'beam_size': beam_size,
        }
        try:
            # Génération des réponses avec le modèle Falcon-7B :
            output = self.monster_client.generate(self.model_name, input_data)
            print("OUTPUT : ", output)
            print("TYPE OUTPUT : ", type(output))
            text = ""
            # Récupération du texte contenant la réponse :
            if isinstance(output, dict):
                text = output.get('text', '')
                print("DICT : ", text)
                print(type(text))
            elif isinstance(output, str):
                text = output
                print("STR : ", output)
                print(type(text))
            # Analyse de la chaîne de caractères pour extraire la réponse :
            if text:
                answer_index = text.find('"answer": "')
                if answer_index != -1:
                    answer_start_index = answer_index + len('"answer": "')
                    answer_end_index = text.find('"', answer_start_index)
                    response = text[answer_start_index:answer_end_index]
                else:
                    response = text
            else:
                response = "Désolé. Je n'ai pas trouvé de réponse à votre question."
            response_translated = self.translate_answer(response)
            return response_translated
        except Exception as e:
            print(f"Une erreur s'est produite : {e}")



    """ Méthode qui traduit la réponse du modèle en français. 
    Elle garantie que la réponse sera en langue française """
    def translate_answer(self, text):
        translator = Translator(to_lang=self.translate_language)
        translation = translator.translate(text)
        return translation