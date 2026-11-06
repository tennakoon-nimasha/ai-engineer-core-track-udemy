import os
import re
from pathlib import Path

import numpy as np
from chromadb import PersistentClient
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential

load_dotenv(override=True)

MODEL = "gemini-3.5-flash-lite"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
SUMMARIES_PATH = Path(__file__).parent.parent / "summaries"

collection_name = "docs"
embedding_model = "all-MiniLM-L6-v2"
wait = wait_exponential(multiplier=1, min=10, max=240)

gemini = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"), base_url=GEMINI_BASE_URL)
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)

# Per sub-query: keep this many chunks after hybrid ranking.
LOCAL_K = 5
# Dense candidate pool size before hybrid re-ranking.
DENSE_K = 20
# Lexical candidate pool size from full-corpus term overlap.
LEXICAL_K = 20
# Cap on merged chunks passed to the final answer.
FINAL_K = 10
# Blend for hybrid score: favors exact names/IDs (helps MRR) while keeping semantics.
DENSE_WEIGHT = 0.6
LEXICAL_WEIGHT = 0.4

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Answer ONLY using the provided Knowledge Base extracts.
If the extracts do not contain the answer, say you don't know.
Do not invent facts. Be accurate, relevant and complete for the user's question.

Knowledge Base extracts:
{context}
"""


class Result(BaseModel):
    page_content: str
    metadata: dict


class SubQuery(BaseModel):
    query: str = Field(description="Short knowledge-base search query")
    # Kept for schema compatibility with earlier planner output, but unused in retrieval.
    # Metadata keyword filtering hurt recall/MRR when ingest keywords were sparse or mismatched.
    keywords: list[str] = Field(
        default_factory=list,
        description="Unused. Previously used for metadata keyword filtering.",
    )


class QueryPlan(BaseModel):
    needs_breakdown: bool = Field(
        description="True when the question needs 2-3 sub-queries to gather separate facts"
    )
    sub_queries: list[SubQuery] = Field(
        description="One sub-query if no breakdown; otherwise 2 or 3 sub-queries"
    )


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9$%]+", text.lower()) if len(token) >= 2]


def lexical_score(query: str, document: str) -> float:
    """Fraction of unique query tokens that appear in the chunk text."""
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    document_lower = document.lower()
    hits = sum(1 for token in query_tokens if token in document_lower)
    return hits / len(query_tokens)


def cosine_similarity(query_vector: list[float], vectors: list[list[float]]) -> np.ndarray:
    query = np.asarray(query_vector, dtype=np.float32)
    matrix = np.asarray(vectors, dtype=np.float32)
    query_norm = np.linalg.norm(query) + 1e-12
    matrix_norms = np.linalg.norm(matrix, axis=1) + 1e-12
    return matrix @ query / (matrix_norms * query_norm)


def _load_collection_rows() -> dict:
    """Load all chunk rows once for hybrid dense + lexical ranking."""
    raw = collection.get(include=["documents", "metadatas", "embeddings"])
    return {
        "ids": raw["ids"],
        "documents": raw["documents"],
        "metadatas": raw["metadatas"],
        "embeddings": raw["embeddings"],
    }


_COLLECTION_ROWS = None


def get_collection_rows() -> dict:
    global _COLLECTION_ROWS
    if _COLLECTION_ROWS is None:
        _COLLECTION_ROWS = _load_collection_rows()
    return _COLLECTION_ROWS


def reload_collection_rows() -> None:
    """Call after re-ingest so retrieval sees the latest collection."""
    global _COLLECTION_ROWS
    _COLLECTION_ROWS = _load_collection_rows()


@retry(wait=wait)
def plan_query(question: str, history=None) -> QueryPlan:
    """Break the question into one or more sub-queries when needed."""
    if history is None:
        history = []

    system_prompt = """
You plan retrieval for an Insurellm knowledge base.
Decide whether the user question needs breakdown into multiple sub-queries.

Rules:
- If the question is a single direct fact, set needs_breakdown=false and return exactly 1 sub-query.
- If the question needs multiple facts (spanning, comparisons, aggregates), set needs_breakdown=true and return 2 or 3 sub-queries.
- Never return more than 3 sub-queries.
- Each sub-query must be a short, concrete search query likely to surface the needed fact.
- You may leave keywords empty; retrieval no longer filters on metadata keywords.
"""
    user_prompt = f"""Conversation history:
{history}

User question:
{question}

Return the retrieval plan.
"""
    response = gemini.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=QueryPlan,
    )
    plan = response.choices[0].message.parsed
    if not plan.sub_queries:
        plan.sub_queries = [SubQuery(query=question)]
    plan.sub_queries = plan.sub_queries[:3]
    return plan


def dense_candidates(query: str, k: int = DENSE_K) -> dict[str, float]:
    """Return chunk_id -> cosine similarity from Chroma dense retrieval."""
    query_vector = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["distances"],
    )
    scores = {}
    for chunk_id, distance in zip(results["ids"][0], results["distances"][0]):
        # Chroma distance is typically L2/cosine distance; convert to a similarity-like score.
        scores[chunk_id] = 1.0 / (1.0 + float(distance))
    return scores


def lexical_candidates(query: str, rows: dict, k: int = LEXICAL_K) -> dict[str, float]:
    """Return chunk_id -> lexical overlap score over the full corpus."""
    scored = []
    for chunk_id, document in zip(rows["ids"], rows["documents"]):
        score = lexical_score(query, document or "")
        if score > 0:
            scored.append((chunk_id, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return {chunk_id: score for chunk_id, score in scored[:k]}


def retrieve_for_subquery(sub_query: SubQuery) -> list[Result]:
    """
    Hybrid retrieval for one sub-query:
    dense similarity + lexical overlap on page_content.

    Metadata keyword filtering is intentionally disabled.
    Reason: hard keyword filters on ingest metadata often over-filtered or mismatched
    the query vocabulary, which lowered recall and MRR. Lexical overlap on the actual
    chunk text recovers exact names, IDs, and dates that dense MiniLM misses.
    """
    rows = get_collection_rows()
    if not rows["ids"]:
        return []

    # --- Disabled: metadata keyword filter-first path ---
    # keywords = normalize_keywords(sub_query.keywords)
    # matched_ids = []
    # if keywords:
    #     for chunk_id, metadata in zip(rows["ids"], rows["metadatas"]):
    #         metadata = metadata or {}
    #         if keyword_overlap(keywords, metadata.get("keywords", "")) > 0:
    #             matched_ids.append(chunk_id)
    # if len(matched_ids) < MIN_CANDIDATES:
    #     return similarity_fallback(sub_query.query)
    # return similarity_over_ids(sub_query.query, matched_ids, rows, LOCAL_K)

    id_to_index = {chunk_id: index for index, chunk_id in enumerate(rows["ids"])}
    dense_scores = dense_candidates(sub_query.query, DENSE_K)
    lex_scores = lexical_candidates(sub_query.query, rows, LEXICAL_K)

    candidate_ids = set(dense_scores) | set(lex_scores)
    if not candidate_ids:
        return []

    dense_max = max(dense_scores.values()) if dense_scores else 1.0
    lex_max = max(lex_scores.values()) if lex_scores else 1.0

    ranked = []
    for chunk_id in candidate_ids:
        dense = dense_scores.get(chunk_id, 0.0) / dense_max
        lexical = lex_scores.get(chunk_id, 0.0) / lex_max
        combined = DENSE_WEIGHT * dense + LEXICAL_WEIGHT * lexical
        ranked.append((chunk_id, combined))

    ranked.sort(key=lambda item: item[1], reverse=True)

    results = []
    for chunk_id, _score in ranked[:LOCAL_K]:
        index = id_to_index[chunk_id]
        results.append(
            Result(
                page_content=rows["documents"][index],
                metadata=rows["metadatas"][index] or {},
            )
        )
    return results


def merge_chunks(*chunk_lists: list[Result], limit: int = FINAL_K) -> list[Result]:
    merged = []
    seen = set()
    for chunk_list in chunk_lists:
        for chunk in chunk_list:
            key = (chunk.metadata.get("source", ""), chunk.page_content)
            if key in seen:
                continue
            seen.add(key)
            merged.append(chunk)
            if len(merged) >= limit:
                return merged
    return merged


def fetch_context(original_question: str, history=None) -> list[Result]:
    """
    Plan sub-queries, retrieve each with hybrid dense+lexical ranking,
    then merge chunks for the final answer / evaluation.
    """
    plan = plan_query(original_question, history)
    print(
        f"needs_breakdown={plan.needs_breakdown}; "
        f"sub_queries={[sq.query for sq in plan.sub_queries]}"
    )
    per_subquery_hits = [
        retrieve_for_subquery(sub_query) for sub_query in plan.sub_queries
    ]
    return merge_chunks(*per_subquery_hits, limit=FINAL_K)


def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {chunk.metadata.get('source', 'unknown')}:\n{chunk.page_content}"
        for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + (history or [])
        + [{"role": "user", "content": question}]
    )


@retry(wait=wait)
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    """
    Answer a question with multi-query hybrid retrieval.
    The final model call always uses the original user question plus merged chunks.
    """
    chunks = fetch_context(question, history)
    messages = make_rag_messages(question, history, chunks)
    response = gemini.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks
