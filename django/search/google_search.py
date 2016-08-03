#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Searches google for a query.
"""

import json
import requests
import readability
from corenlp.models import Document, Sentence
from .models import QueryMap
from datetime import date
from corenlp.util import annotate_document
from . import settings

class GoogleSearcher(object):
    """
    Searches for things on Google.
    """
    SEARCH_URL='https://www.googleapis.com/customsearch/v1'

    def __init__(self, params_file='.config/params.json'):
        with open(params_file) as f:
            self.params = json.load(f)

    def get_as_document(self, link):
        raw = requests.get(link)
        assert raw.status_code == 200, "Couldn't retrieve document ({})".format(raw.status_code)
        doc = readability.Document(raw.text)
        return Document(
            id = link,
            source = link,
            date = date.today(), # TODO(chaganty): use the google search request to get the dates.
            title = doc.title(),
            gloss = doc.summary(),
            )
        return doc

    def __call__(self, query, **kwargs):
        params_ = dict(self.params)
        params_['num'] = 10 # Max allowed >_<
        params_.update(kwargs)
        params_['q'] = query
        response = requests.get(self.SEARCH_URL, params=params_)
        assert response.status_code == 200, "Something went wrong: " + response.text
        results = response.json()
        # Just return the resultant links.
        return [self.get_as_document(it['link']) for it in results['items']]

TEST_QUERY='U.S. Sen. Ron Johnson opposes faster broadband internet in small towns and got nearly $90,000 in campaign contributions from the telecom special interests.'
SEARCHER = GoogleSearcher()

def get_documents_for_query(query):
    if QueryMap.objects.filter(query=query).exists():
        return QueryMap.objects.filter(query=query)
    else:
        results = []
        for doc in SEARCHER(query):
            doc.save()
            q = QueryMap(query=query, doc=doc)
            q.save()
            results.append(q)
    return results

def do_annotate(doc):
    """
    Annotate document with CoreNLP API.
    """
    sentences, mentions = annotate_document(doc)
    Sentence.objects.bulk_create(sentences)

def test_searcher():
    results = searcher(TEST_QUERY)
    assert len(results) == 10, "Didn't get the right number of results."

if __name__ == "__main__":
    test_searcher()
