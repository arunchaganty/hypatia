"""
Utilities to annotate documents with Stanza
"""

from stanza.nlp.corenlp import CoreNLPClient, AnnotationException
from hashlib import sha1
from datetime import date
from .models import Document, Sentence, Mention
from . import settings
import ipdb

CORENLP_CLIENT = CoreNLPClient(settings.ANNOTATOR_ENDPOINT, settings.DEFAULT_ANNOTATORS)

def annotate_document_from_gloss(doc_gloss,
                                 doc_id=None,
                                 corpus_id="default",
                                 source="default",
                                 date=date.today(),
                                 title="",
                                 metadata=""):
    """
    Annotate a document using the CoreNLPClient
    :param doc_gloss Gloss of the document to parse.
    :param doc_id Document id to use. If none, use the has of the document text.
    :returns A list of Sentences and Mentions
    """
    # Create a Document model.
    if doc_id is None:
        doc_id = sha1(doc_gloss).hexdigest()
    doc = Document(doc_id, corpus_id, source, date, title, doc_gloss, metadata)
    sentences, mentions = annotate_document(doc)
    return doc, sentences, mentions

def annotate_document(doc, **kwargs):
    """
    Annotate a document using the CoreNLPClient
    :param doc_gloss Gloss of the document to parse.
    :param doc_id Document id to use. If none, use the has of the document text.
    :returns A list of Sentences and Mentions
    """
    raw = CORENLP_CLIENT.annotate(doc.gloss, **kwargs)
    sentences = [Sentence(
        corpus_id=doc.corpus_id,
        doc=doc,
        sentence_index=s.sentence_index,
        words=s.words,
        lemmas=s.lemmas,
        pos_tags=s.pos_tags,
        ner_tags=s.ner_tags,
        doc_char_begin=[t.character_span[0] for t in s.tokens],
        doc_char_end=[t.character_span[1] for t in s.tokens],
        dependencies=s.depparse,
        gloss=s.text) for s in raw.sentences]
    #mentions = [Mention(
    #    corpus_id,
    #    document,
    #    s,
    #    m.token_begin,
    #    m.token_end,
    #    m.char_begin,
    #    m.char_end,
    #    m.canonical_char_begin,
    #    m.canonical_char_end,
    #    m.ner,
    #    m.best_entity,
    #    m.best_entity_score,
    #    m.alt_entity is None,
    #    m.alt_entity,
    #    m.alt_entity_score,
    #    m.best_entity_score,
    #    m.text) for m in s.mentions() for s in document]

    # Mentions
    return sentences, []
