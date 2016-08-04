#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities
"""

import readability

def clean_html(html_text):
    """
    Returns title and cleaned html.
    """
    doc = readability.Document(html_text)
    return doc.title(), doc.summary()
