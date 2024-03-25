L'objectif de ce projet est de manipuler un modèle d'Embedding des données.
Ce modèle s'appuie sur un LLM interrogé via l'Api Monster et une BDD Vectorielle.

Il rassemble les fonctionnalités suivantes :
- Charger un dataset.
- Sélectionner une catégorie de données dans le dataset.
- Interroger le LLM sur le dataset.

Le Frontend et le Backend pour manipuler ce modèle sont mis à disposition dans les projets suivants :

https://github.com/SydneyJezequel/applicationIABackend
https://github.com/SydneyJezequel/applicationIAFrontend


Commande pour lancer le projet :
uvicorn EmbeddingController:app --reload --workers 1 --host 0.0.0.0 --port 8011


Commande pour installer les dépendances :
pip install -qU \
    transformers==4.30.2 \
    torch \
    einops==0.6.1 \
    accelerate==0.20.3 \
    datasets==2.14.5 \
    chromadb \
    sentence-transformers==2.2.2

