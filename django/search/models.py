from __future__ import unicode_literals

from django.db import models
from corenlp.models import Document

# Create your models here.
class QueryMap(models.Model):
    """
    The query map caches search results from a search query to the set of Document results.
    """
    query = models.TextField(help_text="Search query entered by user")
    document = models.ForeignKey(Document, help_text="Result")

