from django.apps import AppConfig
from django.conf import settings


class MyAppConfig(AppConfig):
    name = 'text_exp_bot'

    def ready(self):
        middleware = settings.MIDDLEWARE
        settings.MIDDLEWARE = [i for i in middleware if i != 'django.middleware.csrf.CsrfViewMiddleware']
