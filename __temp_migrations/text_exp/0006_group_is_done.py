# -*- coding: utf-8 -*-
# Generated by Django 1.11.2 on 2021-05-28 17:25
from __future__ import unicode_literals

from django.db import migrations
import otree.db.models


class Migration(migrations.Migration):

    dependencies = [
        ('text_exp', '0005_auto_20200909_1020'),
    ]

    operations = [
        migrations.AddField(
            model_name='group',
            name='is_done',
            field=otree.db.models.BooleanField(choices=[(True, 'Yes'), (False, 'No')]),
        ),
    ]
