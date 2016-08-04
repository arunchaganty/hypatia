from django.shortcuts import render, redirect
from corenlp.models import Sentence
from search.models import QueryMap, Claim
from functools import reduce
from django.contrib.postgres.search import SearchQuery, SearchRank
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

# Create your views here.
TEST_QUERY='U.S. Sen. Ron Johnson opposes faster broadband internet in small towns and got nearly $90,000 in campaign contributions from the telecom special interests.'

def index(request):
    return render(request, 'index.html', {})

def search(request):
    if 'query' in request.GET:
        search_query = request.GET['query'].strip()
        QueryMap.get(search_query)
        # First, see if there are enough documents that match this query.
        query = reduce(SearchQuery.bitor, map(SearchQuery, search_query.split()))
        results = Sentence.objects.filter(searchable=query).annotate(rank=SearchRank('searchable', query)).order_by('-rank')
        #if results.count() < 2: # Threshold to hit Google.
        #    # If not, search for them!
        #    query = reduce(SearchQuery.bitor, map(SearchQuery, search_query.split()))
        #    results = Sentence.objects.filter(searchable=query).annotate(rank=SearchRank('searchable', query)).order_by('-rank')
    else:
        search_query = TEST_QUERY
        results = []

    return render(request, 'list.html', {'search_query':search_query, 'results':results})


def claims(request):
    """
    Show all current claims.
    """

    claims = Claim.objects.all()
    paginator = Paginator(claims, 10) # 25 -- should be good enough to load

    page = request.GET.get('page')
    try:
        claims = paginator.page(page)
    except PageNotAnInteger:
        # If page is not an integer, deliver first page.
        claims = paginator.page(1)
    except EmptyPage:
        # If page is out of range (e.g. 9999), deliver last page of results.
        claims = paginator.page(paginator.num_pages)

    return render(request, 'list_claims.html', {'claims':claims})

