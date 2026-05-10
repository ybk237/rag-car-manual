# Chatbot RAG pour Documentation Technique Automobile

**Auteur :** Yohann BIKELE — [github.com/ybk237]

Un chatbot basé sur le **Retrieval Augmented Generation (RAG)** permettant de répondre en langage naturel aux questions sur les messages d'avertissement d'une voiture, en s'appuyant exclusivement sur le manuel officiel du véhicule. Construit avec LangChain.

---

## Table des matières

- [Description du projet](#description-du-projet)
- [Concepts clés et algorithmes](#concepts-clés-et-algorithmes)
- [Structure des fichiers](#structure-des-fichiers)
- [Installation](#installation)
- [Configuration](#configuration)
- [Exécution du notebook](#exécution-du-notebook)
- [Exemple de sortie](#exemple-de-sortie)

---

## Description du projet

Les messages d'avertissement d'une voiture doivent être compris rapidement et avec précision — les mal interpréter peut entraîner des dommages coûteux ou une conduite dangereuse. Ce projet construit un chatbot contextuel qui récupère les sections les plus pertinentes du manuel d'une MG ZS et génère une réponse précise et fondée aux questions du conducteur.

Le système met en œuvre le **RAG Fusion**, une stratégie de récupération améliorée qui améliore le rappel par rapport à une récupération à requête unique.

---

## Concepts clés et algorithmes

### Pipeline RAG Fusion

| Étape | Composant | Description |
|-------|-----------|-------------|
| 1 | Génération multi-requêtes | Le LLM transforme 1 question en 4 requêtes diversifiées |
| 2 | Récupération parallèle | Chaque requête interroge indépendamment le vector store |
| 3 | Reciprocal Rank Fusion | Résultats fusionnés et re-classés par score de fusion |
| 4 | Génération de réponse | Le contexte le mieux classé est injecté dans le prompt final |

### Reciprocal Rank Fusion (RRF)

Pour un document $d$ apparaissant au rang $i$ dans la liste de résultats $r$ :

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + i}$$

- $k = 60$ : constante de lissage (réduit la dominance des documents en première position)
- Les documents sont dédupliqués et triés par score de fusion décroissant
- Complexité temporelle : **O(Q · N · log N)** où Q = nombre de requêtes, N = nombre de documents récupérés

### Découpage du texte

`RecursiveCharacterTextSplitter` avec encodeur tiktoken :
- `chunk_size = 300` tokens
- `chunk_overlap = 50` tokens (évite la perte de contexte aux frontières des blocs)

---

## Structure des fichiers

```
rag-car-manual/
├── rag_project.ipynb          # Notebook principal — pipeline RAG Fusion complet
├── README.md                  # README en anglais
├── README.fr.md               # Ce fichier (français)
├── LICENSE                    # Licence MIT
└── data/
    └── mg-zs-warning-messages.html   # Manuel MG ZS (section messages d'avertissement)
```

---

## Installation

**Prérequis :** Python 3.10+

```bash
# Cloner le dépôt
git clone https://github.com/VOTRE_PSEUDO/rag-car-manual.git
cd rag-car-manual

# (Recommandé) Créer et activer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# Installer les dépendances
pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain
pip install langchain_core langchain-google-genai
pip install jupyter
```

---

## Configuration

Ce projet nécessite deux clés API :

| Clé | Utilisation | Où l'obtenir |
|-----|-------------|--------------|
| `GOOGLE_API_KEY` | LLM Gemini + Embeddings | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `LANGCHAIN_API_KEY` | Traçage LangSmith (optionnel) | [smith.langchain.com](https://smith.langchain.com) |

> ⚠️ **Ne mettez pas vos clés API dans le dépôt.** Utilisez un fichier `.env` :

```bash
# .env  (ajouter ce fichier au .gitignore)
GOOGLE_API_KEY=votre_clé_ici
LANGCHAIN_API_KEY=votre_clé_ici
```

Puis chargez-le en tête de notebook :

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Exécution du notebook

```bash
jupyter notebook rag_project.ipynb
```

Exécutez toutes les cellules dans l'ordre. La dernière cellule illustre une requête exemple et affiche la réponse du LLM.

---

## Exemple de sortie

La sortie suivante a été produite en utilisant `data/mg-zs-warning-messages.html` comme corpus, avec la requête :

> *"What are the warnings to care about? (réponse en français)"*

```
À consulter ou agir immédiatement :
  • Défaut du système d'allumage — consulter un réparateur agréé MG immédiatement.
  • Défaut ABS — le système ABS est en panne. Consulter un réparateur agréé MG immédiatement.
  • Température du liquide de refroidissement élevée — arrêter le véhicule dès que possible.
  • Vérifier le moteur — panne grave détectée. Arrêter le véhicule et contacter MG.

À consulter dès que possible :
  • Filtre à particules essence plein — consulter un réparateur agréé MG.
  • Pression d'huile basse — arrêter la voiture et vérifier le niveau d'huile.
  • Défaut moteur — performances et émissions affectées. Contacter MG.
  ...
```

---

## Licence

MIT — voir [LICENSE](LICENSE).
