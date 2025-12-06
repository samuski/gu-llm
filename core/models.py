# core/models.py
from django.db import models

class Segment(models.Model):
    row_id = models.IntegerField(unique=True, db_index=True)  # FAISS row offset -> row_id
    doc_id = models.TextField(db_index=True)
    text   = models.TextField()
