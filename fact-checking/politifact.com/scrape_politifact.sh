#!/bin/bash
PAGE_LIMIT=602 # Obtained through inspection.
for i in `seq $PAGE_LIMIT`; do 
  curl http://politifact.com/truth-o-meter/statements/?page=$i | python3 scrape_politifact.py;
done | csvid --output politifact.tsv

