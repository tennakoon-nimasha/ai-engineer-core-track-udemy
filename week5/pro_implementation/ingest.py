import os
import time
from pathlib import Path

from chromadb import PersistentClient
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI, RateLimitError
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv(override=True)

MODEL = "gemini-3.5-flash-lite"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
collection_name = "docs"
embedding_model = "all-MiniLM-L6-v2"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
AVERAGE_CHUNK_SIZE = 100

# Free-tier Gemini allows ~15 requests/minute. Pause between docs to stay under that.
PAUSE_BETWEEN_DOCS_S = 4.5
MAX_RETRIES = 6

gemini = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"), base_url=GEMINI_BASE_URL)
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)


class Result(BaseModel):
    page_content: str
    metadata: dict


def normalize_keywords(keywords: list[str]) -> list[str]:
    """Lowercase, strip, and drop empties for stable keyword metadata."""
    seen = set()
    normalized = []
    for keyword in keywords:
        value = " ".join(str(keyword).lower().split())
        if value and value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )
    keywords: list[str] = Field(
        description=(
            "High-signal search keywords for this chunk: people names, product names, "
            "contract IDs, dates, dollar amounts, job titles, and short aliases. "
            "Prefer specific entities over generic words like company or salary."
        )
    )

    def as_result(self, document):
        keywords = normalize_keywords(self.keywords)
        metadata = {
            "source": document["source"],
            "type": document["type"],
            "keywords": ",".join(keywords),
        }
        return Result(
            page_content=self.headline
            + "\n\n"
            + self.summary
            + "\n\n"
            + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


def fetch_documents():
    """A homemade version of the LangChain DirectoryLoader"""

    documents = []

    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append(
                    {"type": doc_type, "source": file.as_posix(), "text": f.read()}
                )

    print(f"Loaded {len(documents)} documents")
    return documents


def make_prompt(document):
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide:
- a headline
- a summary
- the original text of the chunk
- keywords: specific entities useful for retrieval (people, products, contract IDs, dates, amounts, titles, short aliases). Include every important entity mentioned in the chunk. Avoid generic words like "company", "document", or "information".

Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


def make_messages(document):
    return [
        {"role": "user", "content": make_prompt(document)},
    ]


def process_document(document, max_retries=MAX_RETRIES):
    messages = make_messages(document)
    for attempt in range(1, max_retries + 1):
        try:
            response = gemini.chat.completions.parse(
                model=MODEL,
                messages=messages,
                response_format=Chunks,
            )
            doc_as_chunks = response.choices[0].message.parsed.chunks
            return [chunk.as_result(document) for chunk in doc_as_chunks]
        except RateLimitError as e:
            # Free tier is 15 requests/minute for this model; wait and retry.
            wait_s = 25
            if getattr(e, "response", None) is not None:
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    wait_s = max(wait_s, int(float(retry_after)) + 1)
            print(
                f"Rate limited on attempt {attempt}/{max_retries}; sleeping {wait_s}s"
            )
            if attempt == max_retries:
                raise
            time.sleep(wait_s)


def create_chunks(documents, pause_s=PAUSE_BETWEEN_DOCS_S):
    """
    Create chunks one document at a time with a pause between calls.
    Stays under Gemini free-tier 15 requests/minute (~1 call every 4s).
    """
    chunks = []
    for i, document in enumerate(tqdm(documents)):
        chunks.extend(process_document(document))
        if i < len(documents) - 1:
            time.sleep(pause_s)
    return chunks


def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(texts)

    collection = chroma.get_or_create_collection(collection_name)

    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
