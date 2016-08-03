from django.shortcuts import render, redirect
from corenlp.models import Sentence
from django.contrib.postgres.search import SearchQuery, SearchRank

# Create your views here.

def index(request):
    return render(request, 'index.html', {})

def search(request):
    print(request.GET)
    if 'query' in request.GET:
        query = request.GET['query'].strip()
        # First, see if there are enough documents that match this query.
        # If not, search for them!
        queries = [SearchQuery(q) for q in query.split()]
        query = queries[0]
        for q in queries[1:]:
            query = query or q
        results = Sentence.objects.filter(searchable=query).annotate(rank=SearchRank('searchable', query)).order_by('-rank')
        print(results.count())
        #results = Sentence.objects.filter(searchable=query)
    else:
        results = []

    return render(request, 'list.html', {'results':results})

