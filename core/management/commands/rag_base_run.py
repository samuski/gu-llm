from django.core.management.base import BaseCommand

from core.rag.config import TOPICS_PATH, BASE_RUN_LOG, FAISS_INDEX
from core.rag.base_run import run_base
from core.rag.retriever import FaissRetriever

from core.models import Segment  # has row_id, doc_id, text

class Command(BaseCommand):

    def handle(self, *args, **opts):
        ST_MODEL_PATH = "sentence-transformers/all-MiniLM-L12-v2"

        retriever = FaissRetriever(index_path=FAISS_INDEX, st_model_path=ST_MODEL_PATH)
        run_base(TOPICS_PATH, BASE_RUN_LOG, retriever, Segment)
        self.stdout.write(self.style.SUCCESS(f"Wrote {BASE_RUN_LOG}"))
