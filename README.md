# RAG Chatbot for Car Technical Documentation

A **Retrieval-Augmented Generation (RAG)** chatbot that answers natural-language questions about car warning messages, grounded strictly in the vehicle's official manual. Built with LangChain.

---

## Table of Contents

- [Project Description](#project-description)
- [Key Concepts & Algorithms](#key-concepts--algorithms)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Notebook](#running-the-notebook)
- [Example Output](#example-output)

---

## Project Description

Car warning messages must be understood quickly and accurately — misinterpreting them can lead to costly damage or unsafe driving. This project builds a context-aware chatbot that retrieves the most relevant sections of an MG ZS car manual and generates a grounded, accurate answer to driver queries.

The system implements **RAG Fusion**, an enhanced retrieval strategy that significantly improves recall compared to naive single-query retrieval.

---

## Key Concepts & Algorithms

### RAG Fusion Pipeline

| Step | Component | Description |
|------|-----------|-------------|
| 1 | Multi-query generation | LLM expands 1 question into 4 diverse queries |
| 2 | Parallel retrieval | Each query runs independently against the vector store |
| 3 | Reciprocal Rank Fusion | Results merged and re-ranked by fusion score |
| 4 | Answer generation | Top-ranked context injected into final LLM prompt |

### Reciprocal Rank Fusion (RRF)

For a document $d$ appearing at rank $i$ in result list $r$:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + i}$$

- $k = 60$: smoothing constant (reduces dominance of rank-1 documents)
- Documents are de-duplicated and sorted by descending fusion score
- Time complexity: **O(Q · N · log N)** where Q = number of queries, N = number of retrieved documents

### Text Chunking

`RecursiveCharacterTextSplitter` with tiktoken encoder:
- `chunk_size = 300` tokens
- `chunk_overlap = 50` tokens (avoids losing context at chunk boundaries)

---

## File Structure

```
rag-car-manual/
├── rag_project.ipynb          # Main notebook — full RAG Fusion pipeline
├── README.md                  # This file (English)
├── README.fr.md               # French README
├── LICENSE                    # MIT License
└── data/
    └── mg-zs-warning-messages.html   # MG ZS car manual (warning messages section)
```

---

## Installation

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rag-car-manual.git
cd rag-car-manual

# (Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# Install dependencies
pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain
pip install langchain_core langchain-google-genai
pip install jupyter
```

---

## Configuration

This project requires two API keys:

| Key | Purpose | Where to obtain |
|-----|---------|-----------------|
| `GOOGLE_API_KEY` | Gemini LLM + Embeddings | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `LANGCHAIN_API_KEY` | LangSmith tracing (optional) | [smith.langchain.com](https://smith.langchain.com) |

> ⚠️ **Do not commit API keys to version control.** Use a `.env` file:

```bash
# .env  (add this file to .gitignore)
GOOGLE_API_KEY=your_key_here
LANGCHAIN_API_KEY=your_key_here
```

Then load it at the top of your notebook:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Running the Notebook

```bash
jupyter notebook rag_project.ipynb
```

Run all cells in order. The final cell demonstrates a sample query and prints the LLM's answer.

---

## Example Output

The following output was produced using `data/mg-zs-warning-messages.html` as the corpus, with the query:

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

## License

MIT — see [LICENSE](LICENSE).
