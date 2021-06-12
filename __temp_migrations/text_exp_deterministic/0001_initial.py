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
                ('set_parameters', otree.db.models.BooleanField(choices=[(True, 'Yes'), (False, 'No')], default=False)),
                ('receiver_choice', otree.db.models.BooleanField(choices=[(True, 'Yes'), (False, 'No')])),
                ('lottery_result', otree.db.models.FloatField(null=True)),
                ('sender_timeout', otree.db.models.BooleanField(choices=[(True, True), (False, False)], default=False)),
                ('receiver_timeout', otree.db.models.BooleanField(choices=[(True, True), (False, False)], default=False)),
                ('sender_payoff', otree.db.models.IntegerField(null=True)),
                ('receiver_payoff', otree.db.models.FloatField(null=True)),
                ('receiver_passed_test', otree.db.models.FloatField(null=True)),
                ('failed_intro_test', otree.db.models.BooleanField(choices=[(True, True), (False, False)], default=False)),
                ('instruction_timeout', otree.db.models.BooleanField(choices=[(True, True), (False, False)], default=False)),
                ('pass_intro_test', otree.db.models.BooleanField(choices=[(True, 'Yes'), (False, 'No')])),
                ('score_0', otree.db.models.FloatField(null=True)),
                ('score_1', otree.db.models.FloatField(null=True)),
                ('score_2', otree.db.models.FloatField(null=True)),
                ('score_3', otree.db.models.FloatField(null=True)),
                ('score_4', otree.db.models.FloatField(null=True)),
                ('score_5', otree.db.models.FloatField(null=True)),
                ('score_6', otree.db.models.FloatField(null=True)),
                ('review_0', otree.db.models.LongStringField(null=True)),
                ('review_1', otree.db.models.LongStringField(null=True)),
                ('review_2', otree.db.models.LongStringField(null=True)),
                ('review_3', otree.db.models.LongStringField(null=True)),
                ('review_4', otree.db.models.LongStringField(null=True)),
                ('review_5', otree.db.models.LongStringField(null=True)),
                ('review_6', otree.db.models.LongStringField(null=True)),
                ('positive_review_0', otree.db.models.LongStringField(null=True)),
                ('positive_review_1', otree.db.models.LongStringField(null=True)),
                ('positive_review_2', otree.db.models.LongStringField(null=True)),
                ('positive_review_3', otree.db.models.LongStringField(null=True)),
                ('positive_review_4', otree.db.models.LongStringField(null=True)),
                ('positive_review_5', otree.db.models.LongStringField(null=True)),
                ('positive_review_6', otree.db.models.LongStringField(null=True)),
                ('negative_review_0', otree.db.models.LongStringField(null=True)),
                ('negative_review_1', otree.db.models.LongStringField(null=True)),
                ('negative_review_2', otree.db.models.LongStringField(null=True)),
                ('negative_review_3', otree.db.models.LongStringField(null=True)),
                ('negative_review_4', otree.db.models.LongStringField(null=True)),
                ('negative_review_5', otree.db.models.LongStringField(null=True)),
                ('negative_review_6', otree.db.models.LongStringField(null=True)),
                ('random_positive_negative_review_0', otree.db.models.LongStringField(null=True)),
                ('random_positive_negative_review_1', otree.db.models.LongStringField(null=True)),
                ('random_positive_negative_review_2', otree.db.models.LongStringField(null=True)),
                ('random_positive_negative_review_3', otree.db.models.LongStringField(null=True)),
                ('random_positive_negative_review_4', otree.db.models.LongStringField(null=True)),
                ('random_positive_negative_review_5', otree.db.models.LongStringField(null=True)),
                ('random_positive_negative_review_6', otree.db.models.LongStringField(null=True)),
                ('average_score', otree.db.models.FloatField(null=True)),
                ('sender_answer_reviews', otree.db.models.LongStringField(null=True)),
                ('sender_answer_negative_reviews', otree.db.models.LongStringField(null=True)),
                ('sender_answer_positive_reviews', otree.db.models.LongStringField(null=True)),
                ('sender_answer_scores', otree.db.models.FloatField(null=True)),
                ('sender_answer_index', otree.db.models.IntegerField(choices=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)], null=True)),
                ('chosen_index', otree.db.models.IntegerField(choices=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)], null=True)),
                ('session', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='text_exp_deterministic_group', to='otree.Session')),
            ],
            options={
                'db_table': 'text_exp_deterministic_group',
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
                ('intro_test', otree.db.models.StringField(default='', max_length=10000, null=True, verbose_name='Do you have any comments on this HIT?')),
                ('dm_test_chosen_review_1', otree.db.models.BooleanField(choices=[(True, 'Yes'), (False, 'No')], verbose_name='Did you see this during the experiment?')),
                ('dm_test_chosen_review_2', otree.db.models.BooleanField(choices=[(True, 'Yes'), (False, 'No')], verbose_name='Did you see this during the experiment?')),
                ('dm_test_not_chosen_review_1', otree.db.models.BooleanField(choices=[(True, 'Yes'), (False, 'No')], verbose_name='Did you see this during the experiment?')),
                ('dm_test_not_chosen_review_2', otree.db.models.BooleanField(choices=[(True, 'Yes'), (False, 'No')], verbose_name='Did you see this during the experiment?')),
                ('group', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='text_exp_deterministic.Group')),
                ('participant', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='text_exp_deterministic_player', to='otree.Participant')),
                ('session', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='text_exp_deterministic_player', to='otree.Session')),
            ],
            options={
                'db_table': 'text_exp_deterministic_player',
            },
            bases=(otree_save_the_change.mixins.SaveTheChange, models.Model),
        ),
        migrations.CreateModel(
            name='Subsession',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('round_number', otree.db.models.PositiveIntegerField(db_index=True, null=True)),
                ('condition', otree.db.models.StringField(max_length=10000, null=True)),
                ('session', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='text_exp_deterministic_subsession', to='otree.Session')),
            ],
            options={
                'db_table': 'text_exp_deterministic_subsession',
            },
            bases=(otree_save_the_change.mixins.SaveTheChange, models.Model),
        ),
        migrations.AddField(
            model_name='player',
            name='subsession',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='text_exp_deterministic.Subsession'),
        ),
        migrations.AddField(
            model_name='group',
            name='subsession',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='text_exp_deterministic.Subsession'),
        ),
    ]
