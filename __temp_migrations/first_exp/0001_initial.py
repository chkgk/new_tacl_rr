# -*- coding: utf-8 -*-
# Generated by Django 1.11.2 on 2020-09-08 12:59
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion
import otree.db.models
import otree_save_the_change.mixins


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('otree', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Group',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('id_in_subsession', otree.db.models.PositiveIntegerField(db_index=True, null=True)),
                ('round_number', otree.db.models.PositiveIntegerField(db_index=True, null=True)),
                ('sender_answer', otree.db.models.FloatField(null=True, verbose_name='Please provide your estimation for the probability of sampling the color red.')),
                ('receiver_choice', otree.db.models.BooleanField(choices=[(True, 'Yes'), (False, 'No')])),
                ('lottery_result', otree.db.models.FloatField(null=True)),
                ('sender_timeout', otree.db.models.BooleanField(choices=[(True, True), (False, False)], default=False)),
                ('receiver_timeout', otree.db.models.BooleanField(choices=[(True, True), (False, False)], default=False)),
                ('sender_payoff', otree.db.models.IntegerField(null=True)),
                ('receiver_payoff', otree.db.models.IntegerField(null=True)),
                ('x_lottery', otree.db.models.FloatField(null=True)),
                ('y_lottery', otree.db.models.FloatField(null=True)),
                ('p_lottery', otree.db.models.FloatField(null=True)),
                ('session', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='first_exp_group', to='otree.Session')),
            ],
            options={
                'db_table': 'first_exp_group',
            },
            bases=(otree_save_the_change.mixins.SaveTheChange, models.Model),
        ),
        migrations.CreateModel(
            name='Player',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('id_in_group', otree.db.models.PositiveIntegerField(db_index=True, null=True)),
                ('_payoff', otree.db.models.CurrencyField(default=0, null=True)),
                ('round_number', otree.db.models.PositiveIntegerField(db_index=True, null=True)),
                ('_gbat_arrived', otree.db.models.BooleanField(choices=[(True, 'Yes'), (False, 'No')], default=False)),
                ('_gbat_grouped', otree.db.models.BooleanField(choices=[(True, 'Yes'), (False, 'No')], default=False)),
                ('name', otree.db.models.StringField(max_length=10000, null=True, verbose_name='What is your name?')),
                ('age', otree.db.models.IntegerField(null=True, verbose_name='What is your age?')),
                ('gender', otree.db.models.StringField(choices=[('Male', 'Male'), ('Female', 'Female')], max_length=10000, null=True, verbose_name='What is your gender?')),
                ('is_student', otree.db.models.BooleanField(choices=[(True, 'Yes'), (False, 'No')], verbose_name='Are you a student?')),
                ('occupation', otree.db.models.StringField(max_length=10000, null=True, verbose_name='What is your occupation?')),
                ('residence', otree.db.models.StringField(max_length=10000, null=True, verbose_name='What is your home town?')),
                ('group', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='first_exp.Group')),
                ('participant', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='first_exp_player', to='otree.Participant')),
                ('session', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='first_exp_player', to='otree.Session')),
            ],
            options={
                'db_table': 'first_exp_player',
            },
            bases=(otree_save_the_change.mixins.SaveTheChange, models.Model),
        ),
        migrations.CreateModel(
            name='Subsession',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('round_number', otree.db.models.PositiveIntegerField(db_index=True, null=True)),
                ('session', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='first_exp_subsession', to='otree.Session')),
            ],
            options={
                'db_table': 'first_exp_subsession',
            },
            bases=(otree_save_the_change.mixins.SaveTheChange, models.Model),
        ),
        migrations.AddField(
            model_name='player',
            name='subsession',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='first_exp.Subsession'),
        ),
        migrations.AddField(
            model_name='group',
            name='subsession',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='first_exp.Subsession'),
        ),
    ]
