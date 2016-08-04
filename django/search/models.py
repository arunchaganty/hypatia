from __future__ import unicode_literals

import logging
from datetime import datetime
import requests
from django.db import models
from corenlp.models import Document
from .util import clean_html
from django.conf import settings

# Create your models here.
class QueryMap(models.Model):
    """
    The query map caches search results from a search query to the set of Document results.
    """
    query = models.TextField(help_text="Search query entered by user")
    document = models.ForeignKey(Document, help_text="Result")

    @classmethod
    def query_google(cls, query, **kwargs):
        """
        Query google for document.
        """
        SEARCH_URL='https://www.googleapis.com/customsearch/v1'
        kwargs.update({'q' : query,
                       'cx' : settings.SEARCH_GOOGLE_CX,
                       'num' : 10, # Maximum allowed by Google
                       'key' : settings.SEARCH_GOOGLE_KEY,})
        response = requests.get(SEARCH_URL, params=kwargs)
        assert response.status_code == 200, "Something went wrong: " + response.text
        results = response.json()
        # Grab the resultant links.
        ret = []
        for result in results['items']:
            link = result['link']
            response = requests.get(link)
            if response.status_code != 200:
                logging.warning("Couldn't retrieve document from %s (returned %d)", link, response.status_code)
                continue
            title, cleaned_html = clean_html(response.text)
            ret.append(Document(
                id = link,
                source = link,
                timestamp = datetime.now(),
                title = title,
                gloss = cleaned_html,
                ))
        return ret

    @classmethod
    def get(cls, query):
        """
        Get query or ask Google for elements.
        """
        if not cls.objects.filter(query=query).exists():
            for doc in cls.query_google(query):
                # Save the document
                doc.save()
                # Annotate the document
                doc.annotate()
                # Create a new QueryMap
                cls(query=query, document=doc).save()
        return cls.objects.filter(query=query)


class Claim(models.Model):
    """
    A set of claims
    """
    timestamp = models.DateTimeField(help_text="Timestamp of this statement")
    source = models.TextField(help_text="Source of the statement")
    claim = models.TextField(help_text="Text of the claim")
    judgement = models.TextField(help_text="Politifact judgement")
    summary = models.TextField(help_text="Explanation")
    url = models.TextField(help_text="Link to politifact page")

