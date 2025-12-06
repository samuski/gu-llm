from django.core.management.base import BaseCommand

from core.models import Segment
from core.rag.config import (
    TOPICS_PATH, FAISS_INDEX,
    GENERATE_RETRIEVE_GENERATE_LOG,
    LLAMA_MODEL_ID, QUERY_EXPAND_MAX_NEW_TOKENS, ANSWER_MAX_NEW_TOKENS
)
from core.rag.retriever import FaissRetriever
from core.rag.query_expander import LlamaQueryExpander
from core.rag.answerer import LlamaAnswerer
from core.rag.generate_retrieve_generate import run_generate_retrieve_generate


class Command(BaseCommand):
    def handle(self, *args, **opts):
        ST_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

        retriever = FaissRetriever(index_path=FAISS_INDEX, st_model_path=ST_MODEL)
        expander = LlamaQueryExpander(model_id=LLAMA_MODEL_ID, max_new_tokens=QUERY_EXPAND_MAX_NEW_TOKENS)
        answerer = LlamaAnswerer(model_id=LLAMA_MODEL_ID, max_new_tokens=ANSWER_MAX_NEW_TOKENS)

        run_generate_retrieve_generate(
            TOPICS_PATH,
            GENERATE_RETRIEVE_GENERATE_LOG,
            retriever,
            expander,
            answerer,
            Segment,
            topk=3,
            limit=None,
        )
        self.stdout.write(self.style.SUCCESS(f"Wrote {GENERATE_RETRIEVE_GENERATE_LOG}"))
