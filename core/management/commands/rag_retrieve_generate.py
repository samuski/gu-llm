from django.core.management.base import BaseCommand

from core.models import Segment
from core.rag.config import TOPICS_PATH, FAISS_INDEX, RETRIEVE_GENERATE_LOG, LLAMA_MODEL_ID, ANSWER_MAX_NEW_TOKENS
from core.rag.retriever import FaissRetriever
from core.rag.retrieve_generate import run_retrieve_generate
from core.rag.answerer import LlamaAnswerer


class Command(BaseCommand):
    def handle(self, *args, **opts):
        ST_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

        retriever = FaissRetriever(index_path=FAISS_INDEX, st_model_path=ST_MODEL)
        answerer = LlamaAnswerer(model_id=LLAMA_MODEL_ID, max_new_tokens=ANSWER_MAX_NEW_TOKENS)

        run_retrieve_generate(TOPICS_PATH, RETRIEVE_GENERATE_LOG, retriever, answerer, Segment, topk=3, limit=None)
        self.stdout.write(self.style.SUCCESS(f"Wrote {RETRIEVE_GENERATE_LOG}"))
