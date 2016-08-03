from django.shortcuts import render, redirect

# Create your views here.

def index(request):
    return render(request, 'index.html', {})

def search(request, query=""):
    query = query.strip()
    if len(query) == 0:
        return redirect("/")



    results = [
        {
            "id" : 0,
            "statement" : "This is a statement",
            "href" : "",},
        {
            "id" : 1,
            "statement" : "This is a statement",
            "href" : "",},
        ]

    return render(request, 'list.html', {'results':results})

