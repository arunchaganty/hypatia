from django.shortcuts import render, redirect
from corenlp.models import Sentence
from functools import reduce
from django.contrib.postgres.search import SearchQuery, SearchRank

# Create your views here.
TEST_QUERY='U.S. Sen. Ron Johnson opposes faster broadband internet in small towns and got nearly $90,000 in campaign contributions from the telecom special interests.'

def index(request):
    return render(request, 'index.html', {})

def search(request):
    if 'query' in request.GET:
        search_query = request.GET['query'].strip()
        # First, see if there are enough documents that match this query.
        # If not, search for them!
        #query = SearchQuery(' | '.join(search_query.split()))
        query = reduce(SearchQuery.bitor, map(SearchQuery, search_query.split()))
        print(query)
        results = Sentence.objects.filter(searchable=query).annotate(rank=SearchRank('searchable', query)).order_by('-rank')
        print(results.count())
        #results = Sentence.objects.filter(searchable=query)
    else:
        search_query = TEST_QUERY
        results = []

    return render(request, 'list.html', {'search_query':search_query, 'results':results})

