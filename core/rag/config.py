from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "msmarco-reduced-trecrag"
TOPICS_PATH = DATA_DIR / "trec-rag-2024" / "topics.rag24.test-reduced.txt"
FAISS_INDEX = DATA_DIR / "index" / "docs.index"

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
BASE_RUN_LOG = LOG_DIR / "base-run.txt"
GENERATE_RETRIEVE_LOG = LOG_DIR / "generate-retrieve.txt"
RETRIEVE_GENERATE_LOG = LOG_DIR / "retrieve-generate.txt"
GENERATE_RETRIEVE_GENERATE_LOG = LOG_DIR / "generate-retrieve-generate.txt"

TOP_K = 3
LIMIT = 50
CANDIDATE_K = 50 # For reranking

MAX_TEXT_CHARS = None # no clipping
QUERY_EXPAND_MAX_NEW_TOKENS = 64
ANSWER_MAX_NEW_TOKENS = 256

LLAMA_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"