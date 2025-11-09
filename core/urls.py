
from django.contrib import admin
from django.urls import path

from django.urls import path
from core.views import chat
    
urlpatterns = [
    path('admin/', admin.site.urls),
    path('chat/', chat, name='chat')
]

from django.views.generic import TemplateView
urlpatterns += [ path("", TemplateView.as_view(template_name="index.html")) ]