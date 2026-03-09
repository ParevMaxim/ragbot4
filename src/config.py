import os
from dotenv import load_dotenv

load_dotenv()


def default_data_dir() -> str:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        base = os.path.expanduser("~")
    return os.path.join(base, "RAGChat")


def _env_bool(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# LLM models
CHAT_MODEL_MAIN = os.getenv("CHAT_MODEL_MAIN") or os.getenv("CHAT_MODEL") or "llama3.1"
CHAT_MODEL_SECONDARY = os.getenv("CHAT_MODEL_SECONDARY", CHAT_MODEL_MAIN)
AGGREGATE_MODEL = os.getenv("AGGREGATE_MODEL", CHAT_MODEL_MAIN)
REWRITE_MODEL = os.getenv("REWRITE_MODEL", CHAT_MODEL_MAIN)

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# Documentation language
DOC_LANGUAGE = os.getenv("DOC_LANGUAGE", "same")

# Data dirs
KB_DIR = os.getenv("KB_DIR", os.path.join(default_data_dir(), "kb"))

# Pipeline switches
ENABLE_QUERY_REWRITE = _env_bool("ENABLE_QUERY_REWRITE", "0")
ENABLE_FACTOID_BOOST = _env_bool("ENABLE_FACTOID_BOOST", "1")
ENABLE_MMR = _env_bool("ENABLE_MMR", "1")

# Answer mode
ANSWER_WITH_CITATIONS = _env_bool("ANSWER_WITH_CITATIONS", "1")
ENABLE_YEAR_GUARD = _env_bool("ENABLE_YEAR_GUARD", "1")

# Index-time cleaning (optional)
CLEAN_BRACKETED_NUM_REFS = _env_bool("CLEAN_BRACKETED_NUM_REFS", "0")

# Ollama generation options
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "256"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "4096"))

OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")

LLM_OPTIONS = {
    "num_predict": LLM_NUM_PREDICT,
    "temperature": LLM_TEMPERATURE,
    "top_p": LLM_TOP_P,
    "num_ctx": LLM_NUM_CTX,
}