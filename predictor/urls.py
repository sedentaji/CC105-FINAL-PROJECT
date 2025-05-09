# predictor/urls.py
from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('predict/', views.predict, name='predict'),
    path('dashboard/', views.dashboard, name='dashboard'),
]