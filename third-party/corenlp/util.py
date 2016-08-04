"""
Utilities to annotate documents with Stanza
"""

from stanza.nlp.corenlp import CoreNLPClient
from django.conf import settings

class CoreNLPService(object):
    """
    A singleton CoreNLP server.
    """
    instance = None
    def __init__(self):
        if CoreNLPService.instance is None:
            CoreNLPService.instance = CoreNLPClient(settings.CORENLP_ANNOTATOR_ENDPOINT, settings.CORENLP_DEFAULT_ANNOTATORS)

    def __getattr__(self, attr):
        return getattr(CoreNLPService.instance, attr)

