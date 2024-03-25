from monsterapi import client
from translate import Translator

import config


# URL de l'Api : https://monsterapi.ai/user/playground?model=falcon-7b-instruct
class LlmModel:
    """ Modèle LLM (Llama2-7B ou Falcon7B) """



    def __init__(self):
        """ Constructeur """
        # Définition du modèle :
        self.model_name = config.MODEL_NAME_FALCON7B
        # Initialisation des pipeline, tokenizer et modèle :
        self.monster_client = self.initialize_model()
        # Sélection de la langue :
        self.prompt_language = config.PROMPT_LANGUAGE
        self.translate_language = config.TRANSLATE_LANGUAGE



    def initialize_model(self):
        """ Méthode qui initialise le modèle """
        # Initialisation du client MonsterAPI avec la clé API :
        monster_api_key = config.MONSTER_API_KEY
        monster_client = client(monster_api_key)
        # Renvoi du pipeline, du tokenizer et du client MonsterAPI :
        return monster_client



    def set_language(self, language):
        """ Méthode qui sélectionne la langue des réponses """
        self.prompt_language = language



    def generate_answer(self, question):
        """ Méthode qui interroge le modèle Falcon7B """
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



    def generate_enriched_answer(self, question, context=None):
        """ Méthode qui interroge le modèle Falcon7B en y ajoutant un contexte """
        # Préparation du prompt :
        prompt = f'"question": "{question}", "context": "{context}", "language": "français"' if context is not None else f'"question": "{question}", "context": "", "language": "français"'
        # Initialisation des paramètre de l'input :
        top_k = 15
        top_p = 0.1
        temp = 0.1
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



    def translate_answer(self, text):
        """ Méthode qui traduit la réponse du modèle en français """
        translator = Translator(to_lang=self.translate_language)
        translation = translator.translate(text)
        return translation

