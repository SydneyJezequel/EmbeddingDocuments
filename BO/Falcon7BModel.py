import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM






""" Modèle Falcon7B """
class Falcon7BModel:



    """ Constructeur """
    def __init__(self):
        # Définition du modèle :
        model_name = "tiiuae/falcon-7b-instruct"
        # Initialisation des pipeline, tokenizer et modèle :
        self.pipeline, self.tokenizer = self.initialize_model(model_name)



    """ Méthode qui initialise le modèle """
    def initialize_model(self, model_name):
        # Initialisation du Tokenizer :
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Configuration du pipeline pour de la génération de texte :
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        # Renvoi du pipeline et du tokenizer :
        return pipeline, tokenizer



    """ Méthode qui génère une réponse """
    def generate_answer(self, question, context=None):
        # Préparation du prompt :
        prompt = question if context is None else f"{context}\n\n{question}"
        # Génération des réponses :
        sequences = self.pipeline(
            prompt,
            max_length=500,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # Extraction et renvoi du texte généré :
        return sequences['generated_text']

