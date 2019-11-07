from . import views
from django.urls import path

urlpatterns = [
    path('', views.index, name='index'),
    path('corr', views.corr, name='corr'),
    path('c3', views.cthree, name='c3'),
    path('api/corr_data', views.corr_data, name='corr_data'),
    path('api/get_data', views.get_data, name='get_data'),
    path('api/scatter', views.scatter_plot, name='scatter_plot')
]