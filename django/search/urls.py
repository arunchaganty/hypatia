from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'claims/', views.claims),
    url(r'^$', views.search),
]
