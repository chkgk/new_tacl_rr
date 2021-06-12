from otree.api import (
    models, widgets, BaseConstants, BaseSubsession, BaseGroup, BasePlayer,
    Currency as c
)
import pandas as pd
import random
import os
import math


base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, 'first_fix_prob_exp')
problems_data_file_path = os.path.join(data_directory, 'fix_problems.csv')
image_directory = os.path.join(base_directory, '_static', 'first_fix_prob_exp')

author = 'Reut Apel'

doc = """
"""


class Constants(BaseConstants):
    name_in_url = 'two_players_experiment'
    players_per_group = 2
    num_rounds = 50
    sender_payoff_per_round = 1
    seconds_wait = 7
    seconds_wait_first_rounds = 20
    first_rounds = [1, 2, 3, 4, 5]
    num_rounds_long_wait = first_rounds[-1]
    bonus = 1
    # participation_fee = 2.5

    # num_timeouts_remove = 5
    minutes_to_timeout_wait_page = 10
    pay_by_time_for_waiting = 0.09

    problems = pd.read_csv(problems_data_file_path, header=0).sample(frac=1).reset_index(drop=True)


class Subsession(BaseSubsession):

    def creating_session(self):
        """
        This function will run at the beginning of each session
        and will initial the problems parameters for all the rounds
        Each row will be the parameters of the index+1 round (index starts at 0)
        :return:
        """
        if self.round_number == 1:
            for g in self.get_groups():
                if g.set_parameters:
                    print('already created problems for group')
                    continue

                for p in g.get_players():
                    if 'problem_parameters' in p.participant.vars:
                        print('already created problems for player with id {p.id_in_group}')
                        continue

                    if p.id_in_group == 1:  # create the parameters only for experts:
                        print(f'creating session for player with role {p.role()},'
                              f'id_in_subsession: {p.id_in_subsession}')
                        # Load problems and shuffle the them so each subject will get them in a different order
                        problems = Constants.problems
                        problems.X = problems.X.astype(float)
                        problems.Y = problems.Y.astype(float)
                        # Create problem set:
                        p.participant.vars['problem_parameters'] = problems
                        print(f'set parameters to True')
                        g.set_parameters = True

                        # for debug:
                        print('all problems in creating_session: ', p.participant.vars['problem_parameters'])

                        # calculate the best and worst cases for receiver, and the probability to get bonus
                        worst_case_receiver = problems['Y'].sum()

                        self.session.vars['initial_points_receiver'] = math.fabs(worst_case_receiver)
                        # this will be calculated during the experiment,
                        # and will be the sum of the non-negative payoffs in the game: 0- if Y is the result in the lottery
                        # X- if X is the result --> this is the real maximum points from this experiment
                        p.participant.vars['max_points'] = math.fabs(worst_case_receiver)

                        # for other_p in p.get_others_in_group():
                        #     if 'problem_parameters' in other_p.participant.vars:
                        #         print('already created problems for other_p')
                        #         continue
                        #     if other_p.id_in_group == 2:
                        #         print(f'setting parameters for player with id in group {other_p.id_in_group}')
                        #         other_p.participant.vars['problem_parameters'] = problems
                        #         other_p.participant.vars['max_points'] = math.fabs(worst_case_receiver)

        return


class Group(BaseGroup):
    set_parameters = models.BooleanField(initial=False)
    sender_answer = models.FloatField(
        verbose_name='Please provide your estimation for the probability of sampling the color red.',
                     # '\nA probability is a number between 0 and 1',
        min=0.0, max=1.0)
    receiver_choice = models.BooleanField()  # True (1) for Certainty and False (0) for Lottery
    lottery_result = models.FloatField()
    sender_timeout = models.BooleanField(choices=[True, False], initial=False)  # True if the sender had timeout
    receiver_timeout = models.BooleanField(choices=[True, False], initial=False)  # True if the receiver had timeout
    sender_payoff = models.IntegerField()
    receiver_payoff = models.FloatField()
    # receiver_payoff = models.FloatField()
    x_lottery = models.FloatField()
    y_lottery = models.FloatField()
    p_lottery = models.FloatField()
    ev_lottery = models.FloatField()

    def set_payoffs(self):
        print('receiver choice:', self.receiver_choice)
        sender = self.get_player_by_role('Expert')
        receiver = self.get_player_by_role('Decision Maker')
        round_parameters = sender.participant.vars['problem_parameters'].loc[self.round_number-1]
        self.lottery_result = lottery(round_parameters)
        self.x_lottery = round_parameters['X']
        self.y_lottery = round_parameters['Y']
        self.p_lottery = round_parameters['P']
        self.ev_lottery = round_parameters['E']
        if self.lottery_result > float(0):  # if the lottery result is non negative - add it to max_points
            sender.participant.vars['max_points'] += self.lottery_result
        if self.receiver_choice:  # if the receiver choose A: Certainty - both get 0
            sender.payoff = c(0)
            receiver.payoff = c(0.0)
            self.receiver_payoff = 0.0
            self.sender_payoff = 0
            # self.receiver_payoff = float(0)
            # print('self.receiver_choice:', self.receiver_choice, 'self.receiver_payoff:', self.receiver_payoff,
            #       'self.lottery_result:', self.lottery_result)

        # if the receiver choose B: Lottery - sender get 1 point, and receiver get the result of the lottery
        else:
            receiver.payoff = c(self.lottery_result)
            self.receiver_payoff = self.lottery_result
            # self.receiver_payoff = self.lottery_result
            # print('self.receiver_choice:', self.receiver_choice, 'self.receiver_payoff:', self.receiver_payoff,
            #       'self.lottery_result:', self.lottery_result)

            # if there was timeout for the sender --> not get paid
            if self.sender_timeout:
                sender.payoff = c(0)
                self.sender_payoff = 0
            else:
                sender.payoff = c(Constants.sender_payoff_per_round)
                self.sender_payoff = Constants.sender_payoff_per_round

        print('receiver.payoff:', receiver.payoff, 'sender.payoff:', sender.payoff)

        return

    # def is_player_drop_out(self):
    #     """
    #     :return: This function return True if there are less then 2 players in the group - one player drop out,
    #     and False otherwise
    #     """
    #     players_with_us = self.get_players()
    #     return True if len(players_with_us) != Constants.players_per_group else False


def lottery(round_parameters):
    return round_parameters['X'] if random.random() <= round_parameters['P'] else round_parameters['Y']


class Player(BasePlayer):
    name = models.StringField(
        verbose_name='What is your name?'
    )
    age = models.IntegerField(
        verbose_name='What is your age?',
        min=5
    )
    gender = models.StringField(
        choices=['Male', 'Female'],
        verbose_name='What is your gender?',
        widget=widgets.RadioSelect)
    is_student = models.BooleanField(
        verbose_name='Are you a student?',
        widget=widgets.RadioSelect)
    occupation = models.StringField(
        verbose_name='What is your occupation?'
    )
    residence = models.StringField(
        verbose_name='What is your home town?'
    )

    def role(self):
        return {1: 'Expert', 2: 'Decision Maker'}[self.id_in_group]

    # def is_displayed(self):
    #     return self.participant.vars['num_timeout'] < 5


class Session:
    num_participants = 2
