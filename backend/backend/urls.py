
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('userapp.urls')),
    path('irrigation/', include('irrigationApp.urls')),
    path('feedback/', include('feedbackApp.urls')),
]
