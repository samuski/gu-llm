from django.core.management.base import BaseCommand

from core.models import Segment
from core.rag.config import TOPICS_PATH, FAISS_INDEX, GENERATE_RETRIEVE_LOG, LLAMA_MODEL_ID, QUERY_EXPAND_MAX_NEW_TOKENS
from core.rag.retriever import FaissRetriever
from core.rag.generate_retrieve import run_generate_retrieve
from core.rag.query_expander import LlamaQueryExpander


class Command(BaseCommand):
    def handle(self, *args, **opts):
        ST_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

        retriever = FaissRetriever(index_path=FAISS_INDEX, st_model_path=ST_MODEL)
        expander = LlamaQueryExpander(model_id=LLAMA_MODEL_ID, max_new_tokens=QUERY_EXPAND_MAX_NEW_TOKENS)

        run_generate_retrieve(TOPICS_PATH, GENERATE_RETRIEVE_LOG, retriever, expander, Segment)
        self.stdout.write(self.style.SUCCESS(f"Wrote {GENERATE_RETRIEVE_LOG}"))
