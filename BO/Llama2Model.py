import replicate
import os






""" Modèle Llama2Model """
class Llama2Model:



    """ Constructeur """
    def __init__(self, token):
        self.api = self.api_config(token)



    """ Méthode qui configure l'api du modèle Llama2 """
    def api_config(self, token):
        os.environ["REPLICATE_API_TOKEN"] = token
        api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
        return api



    """ Méthode qui interroge le modèle Llama2 """
    def generate_answer(self, question):
        # Attributs :
        result = ""
        # Traitement :
        # Interrogation du modèle :
        output = self.api.run(
            "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            input={"prompt": question}
        )
        # Récupération de la réponse :
        for item in output:
            result += item
        return result



    """ Méthode qui interroge le modèle Llama2 en y ajoutant un contexte """
    def generate_enriched_answer(self, question, context=None):
        # Préparation du prompt :
        prompt = question if context is None else f"{context}\n\n{question}"
        # Génération des réponses avec le modèle Llama 2 via Replicate
        enriched_answer = self.generate_answer(prompt)

        """
        # Génération des réponses avec le modèle local
        sequences_local = self.pipeline(
            prompt,
            max_length=500,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Affichage des résultats :
        print(f"Local Generated Answer: {sequences_local['generated_text']}")
        print(f"Enriched Answer: {enriched_answer}")

        # Retour de la réponse générée localement (ou toute autre logique que vous préférez)
        return sequences_local['generated_text']
        """
        return enriched_answer

