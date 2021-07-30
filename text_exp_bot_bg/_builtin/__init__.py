# This file is auto-generated.
# It's used to aid autocompletion in code editors.

import otree.api
from .. import models


class Page(otree.api.Page):
    def z_autocomplete(self):
        print('Page class!!z_autocomplete')
        self.subsession = models.Subsession()
        self.session = models.Session()
        self.group = models.Group()
        self.player = models.Player()


class WaitPage(otree.api.WaitPage):
    def z_autocomplete(self):
        print('WaitPage class!!z_autocomplete')
        self.subsession = models.Subsession()
        self.group = models.Group()


class Bot(otree.api.Bot):
    def z_autocomplete(self):
        print('Bot class!!z_autocomplete')
        self.subsession = models.Subsession()
        self.group = models.Group()
        self.player = models.Player()
