"""
Utilities to annotate documents with Stanza
"""

from stanza.nlp.corenlp import CoreNLPClient, AnnotationException
from hashlib import sha1
from datetime import date
from .models import Document, Sentence, Mention
from . import settings

CORENLP_CLIENT = CoreNLPClient(settings.ANNOTATOR_ENDPOINT, settings.DEFAULT_ANNOTATORS)

def annotate_document(doc_gloss,
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
    raw = CORENLP_CLIENT.annotate(doc_gloss)

    # Create a Document model.
    if doc_id is None:
        doc_id = sha1(raw.text).hexdigest()

    document = Document(doc_id, corpus_id, source, date, title, raw.text, metadata)
    sentences = [Sentence(
        corpus_id,
        document,
        s.sentence_index,
        s.words,
        s.lemmas,
        s.pos_tags,
        s.ner_tags,
        s.character_begins,
        s.character_ends,
        s.depparse,
        s.text) for s in document]
    mentions = [Mention(
        corpus_id,
        document,
        s,
        m.token_begin,
        m.token_end,
        m.char_begin,
        m.char_end,
        m.canonical_char_begin,
        m.canonical_char_end,
        m.ner,
        m.best_entity,
        m.best_entity_score,
        m.alt_entity is None,
        m.alt_entity,
        m.alt_entity_score,
        m.best_entity_score,
        m.text) for m in s.mentions() for s in document]

    return document, sentences, mentions
