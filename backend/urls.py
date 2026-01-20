from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/users/', include('users.urls')), 
    path('api/chat/', include('chatbot.urls')),
    path('api/', include('markets.urls')),
    path('api/analysis/', include('analysis.urls')),
]
