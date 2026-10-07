# RAGPalette

**Une librairie Python modulaire pour assembler des pipelines de Retrieval-Augmented Generation (RAG).**

RAGPalette relie vos documents à un générateur de réponses : elle découpe les textes, calcule leurs représentations vectorielles (*embeddings*), retrouve les passages pertinents et les transmet à un modèle de langage. Le résultat contient la réponse et les passages utilisés comme contexte.

Son architecture permet de composer un pipeline avec des composants indépendants et de remplacer le découpage, les embeddings, le stockage, la recherche ou la génération selon les besoins du projet.

**Version actuelle : `0.1.0` — projet en développement.**

## Fonctionnalités disponibles

- **Découpage des documents** avec `FixedSizeChunker`, par nombre de caractères et avec chevauchement configurable.
- **Indexation** avec `Indexer`, qui orchestre le découpage, les embeddings et le stockage.
- **Embeddings** avec `SentenceTransformerEmbedder`, ou votre propre implémentation.
- **Stockage vectoriel en mémoire** avec `InMemoryVectorStore` et recherche par similarité cosinus.
- **Recherche dense** avec `DenseRetriever`.
- **Génération** avec `LLMGenerator`, qui accepte une fonction Python prenant un prompt et renvoyant une chaîne de caractères.
- **RAG classique** avec `ClassicRAG`.
- **RAG avec reclassement** avec `RerankedRAG` et `ScoringReranker`, pour réévaluer les passages avant la génération.
- **Contexte inspectable** : chaque résultat de recherche expose son passage, son score et sa source.

## Fonctionnement

```text
Indexation
Documents → Découpage → Embeddings → Stockage vectoriel

RAG classique
Question → Embedding → Recherche → Contexte → Génération → Réponse

RAG avec reclassement
Question → Embedding → Recherche → Reclassement → Contexte → Génération → Réponse
```

Le générateur construit un prompt qui demande au modèle de répondre uniquement à partir du contexte et d'indiquer lorsque celui-ci est insuffisant. Le respect de cette consigne dépend du modèle connecté.

## Installation

Prérequis : **Python 3.10 ou supérieur**.

Depuis la racine du dépôt, après l'avoir cloné ou téléchargé :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Sous Windows, remplacez la commande d'activation par `.venv\Scripts\activate` dans l'invite de commandes, ou `.venv\Scripts\Activate.ps1` dans PowerShell.

Pour utiliser `SentenceTransformerEmbedder`, installez également sa dépendance :

```bash
python -m pip install sentence-transformers
```

Cette dépendance n'est pas déclarée dans le `pyproject.toml` actuel. Elle n'est pas nécessaire pour l'exemple ci-dessous, qui utilise un embedder de démonstration.

## Démarrage rapide

Cet exemple s'exécute sans clé API ni téléchargement de modèle. Les embeddings reposent sur quelques mots-clés et le générateur renvoie le passage sélectionné : ils servent à montrer le fonctionnement du pipeline. Pour une recherche sémantique et des réponses rédigées, connectez ensuite un modèle d'embeddings et un LLM.

Enregistrez le code dans `exemple.py`, puis exécutez `python exemple.py` :

```python
from ragpalette.chunking.fixed import FixedSizeChunker
from ragpalette.core import Document
from ragpalette.generation.generator import LLMGenerator
from ragpalette.ingestion.indexer import Indexer
from ragpalette.retrieval.dense import DenseRetriever
from ragpalette.stores.memory import InMemoryVectorStore
from ragpalette.strategies import ClassicRAG


class DemoEmbedder:
    """Embeddings par mots-clés, uniquement pour cette démonstration."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return [
            float(keyword in query.lower())
            for keyword in ("passe", "remboursement", "livraison")
        ]


def demo_llm(prompt: str) -> str:
    """Renvoie le texte du premier passage, sans appel à un LLM."""
    context = prompt.split("Context:\n", 1)[1].split("\n\nQuestion:", 1)[0]
    if not context.strip():
        return "Le contexte est insuffisant pour répondre."
    first_source = context.split("\n\n", 1)[0]
    return first_source.split("\n", 1)[1]


embedder = DemoEmbedder()
store = InMemoryVectorStore()
indexer = Indexer(
    chunker=FixedSizeChunker(chunk_size=500, overlap=50),
    embedder=embedder,
    vector_store=store,
)

documents = [
    Document(
        id="compte",
        text="Pour réinitialiser votre mot de passe, utilisez la page de récupération.",
        metadata={"source": "aide/compte.txt"},
    ),
    Document(
        id="facturation",
        text="Pour demander un remboursement, contactez le service de facturation.",
        metadata={"source": "aide/facturation.txt"},
    ),
]

for document in documents:
    indexer.index_document(document)

retriever = DenseRetriever(embedder=embedder, vector_store=store)
generator = LLMGenerator(llm=demo_llm)
rag = ClassicRAG(retriever=retriever, generator=generator, top_k=1)

result = rag.run("Comment réinitialiser mon mot de passe ?")
print(result.answer)

for item in result.context:
    print(f"Source : {item.source} | Score : {item.score:.2f}")
```

Sortie attendue :

```text
Pour réinitialiser votre mot de passe, utilisez la page de récupération.
Source : aide/compte.txt | Score : 1.00
```

## Utiliser un modèle d'embeddings

Après avoir installé `sentence-transformers`, vous pouvez remplacer `DemoEmbedder` par `SentenceTransformerEmbedder`. Définissez la variable d'environnement `RAGPALETTE_EMBEDDING_MODEL` avec le nom ou le chemin du modèle de votre choix :

```python
import os

from ragpalette.embeddings.sentence_transformer import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder(
    model_name=os.environ["RAGPALETTE_EMBEDDING_MODEL"]
)
```

Ce composant utilise `encode_document` et `encode_query` avec normalisation des embeddings. Votre version de Sentence Transformers doit exposer ces méthodes. Le chargement d'un modèle distant peut nécessiter un téléchargement lors de la première utilisation.

Utilisez le même embedder pour l'indexation et la recherche. Si vous changez de modèle, recréez le stockage et réindexez vos documents avant de poser des questions.

## Connecter votre LLM

`LLMGenerator` attend une fonction synchrone de signature `llm(prompt: str) -> str`. Si votre client propose déjà une méthode `generate(prompt)` qui renvoie le texte de la réponse, l'intégration prend cette forme :

```python
# mon_client est votre client LLM déjà configuré.
generator = LLMGenerator(llm=mon_client.generate)
rag = ClassicRAG(retriever=retriever, generator=generator, top_k=3)
result = rag.run("Votre question")
```

Si votre client renvoie un objet structuré, écrivez une fonction qui effectue l'appel et en extrait le texte. La configuration du client et ses éventuels identifiants restent à votre charge.

## Ajouter un reclassement des résultats

`RerankedRAG` récupère jusqu'à `candidate_k` passages, les reclasse, puis conserve les `top_k` premiers pour la génération. À partir des composants du démarrage rapide :

```python
from ragpalette.reranking import ScoringReranker
from ragpalette.strategies import RerankedRAG


def score_passage(query: str, text: str) -> float:
    # Score lexical de démonstration ; remplacez-le par votre fonction de scoring.
    query_words = set(query.lower().split())
    passage_words = set(text.lower().split())
    return float(len(query_words & passage_words))


rag = RerankedRAG(
    retriever=retriever,
    reranker=ScoringReranker(scorer=score_passage),
    generator=generator,
    candidate_k=10,
    top_k=1,
)
result = rag.run("Comment réinitialiser mon mot de passe ?")
print(result.answer)
```

Un score plus élevé indique une meilleure pertinence. `top_k` doit être strictement positif et `candidate_k` doit être supérieur ou égal à `top_k`. Dans le contexte renvoyé par `RerankedRAG`, les scores sont ceux du reclassement.

## Personnaliser le pipeline

Les interfaces sont définies dans [`ragpalette/core/protocols.py`](ragpalette/core/protocols.py). Vos composants peuvent les respecter sans hériter d'une classe de base :

| Composant | Méthodes attendues |
| --- | --- |
| `Chunker` | `split(document)` |
| `Embedder` | `embed_documents(texts)`, `embed_query(query)` |
| `VectorStore` | `add_chunks(chunks, embeddings)`, `search(query_embedding, top_k)` |
| `Retriever` | `retrieve(query, top_k)` |
| `Reranker` | `rerank(query, results)` |
| `Generator` | `generate(query, context)` |

Les modèles `Document`, `Chunk`, `SearchResult` et `RAGResult` sont disponibles dans `ragpalette.core`. Les documents acceptent des métadonnées ; la clé `source` sert à identifier l'origine des passages. En son absence, le stockage utilise l'identifiant du document.

## État du projet

| Stratégie | État actuel |
| --- | --- |
| `ClassicRAG` | Implémentée |
| `RerankedRAG` | Implémentée |
| `MultiQueryRAG` | Squelette à implémenter |
| `QueryExpansionRAG` | Squelette à implémenter |
| `HierarchicalRAG` | Squelette à implémenter |
| `SelfReflectiveRAG` | Squelette à implémenter |
| `AgenticRAG` | Squelette à implémenter |
| `GraphRAG` | Squelette à implémenter |

Les stratégies au stade de squelette lèvent actuellement `NotImplementedError` lorsque leur méthode `run` est appelée.

Le stockage fourni conserve les données uniquement en mémoire et parcourt les vecteurs pour chaque recherche. Le dépôt ne fournit pas encore de stockage persistant ni de lecteurs PDF, Word ou HTML : préparez vos textes avant de créer vos objets `Document`.

## Tests

Depuis la racine du dépôt :

```bash
python -m pip install pytest
python -m pytest tests/
```

Les tests existants couvrent notamment la recherche vectorielle, le pipeline classique et le pipeline avec reclassement.

## Contributions et collaboration

Les contributions sont les bienvenues : nouvelles stratégies RAG, composants d'indexation, intégrations de modèles, stockage persistant, tests et documentation. Pour proposer une amélioration, ouvrez une issue ou une pull request sur le dépôt GitHub.

Vous souhaitez collaborer, échanger sur le RAG ou contribuer au projet ? Contactez **Oussama Kaddouri** :

- **Email :** [kaddourioussama189@gmail.com](mailto:kaddourioussama189@gmail.com)
- **Téléphone :** [+212634350272](tel:+212634350272)
