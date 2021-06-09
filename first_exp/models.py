from otree.api import (
    models, widgets, BaseConstants, BaseSubsession, BaseGroup, BasePlayer,
    Currency as c
)
import pandas as pd
import random
import os
import math


base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, 'first_exp')
problems_data_file_path = os.path.join(data_directory, 'problems_data_all_comb.csv')
image_directory = os.path.join(base_directory, '_static', 'first_exp')

author = 'Reut Apel'

doc = """
Your app description
"""


class Constants(BaseConstants):
    name_in_url = 'my_trust'
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


class Subsession(BaseSubsession):

    def creating_session(self):
        """
        This function will run at the beginning of each session
        and will initial the problems parameters for all the rounds
        Each row will be the parameters of the index+1 round (index starts at 0)
        :return:
        """
        if self.round_number == 1:
            problems = pd.read_csv(problems_data_file_path, usecols=['X', 'Y', 'P', 'E'], header=0)
            # Create problem set:
            # get random problem from problems set
            problem_parameters = problems.sample(n=Constants.num_rounds)
            # remove all the problems with the same X and Y, so the players will not get the same one in the experiment
            problem_parameters = problem_parameters.drop_duplicates(subset=['X', 'Y'])
            while problem_parameters.shape[0] != Constants.num_rounds:  # if I drop problems, sample more
                num_problems_miss = Constants.num_rounds - problem_parameters.shape[0]
                # sample num_problems_miss problems
                temp = problems.sample(n=num_problems_miss)
                problem_parameters = problem_parameters.append(temp)
                # remove all the problems with the same X and Y,
                # so the players will not get the same one in the experiment
                problem_parameters = problem_parameters.drop_duplicates(subset=['X', 'Y'])

            problem_parameters = problem_parameters.reset_index(drop=True)
            problem_parameters.X = problem_parameters.X.astype(int)
            problem_parameters.Y = problem_parameters.Y.astype(int)
            self.session.vars['problem_parameters'] = problem_parameters

            # for debug:
            print('all problems in creating_session: ', self.session.vars['problem_parameters'])

            # parameters to check if a player need to be removed, when he had timeout more than num_timeouts_remove
            # self.session.vars['remove_player'] = False
            # self.session.vars['player_to_remove'] = None
            # self.session.vars['player_not_remove'] = None
            # calculate the best and worst cases for receiver, and the probability to get bonus
            worst_case_receiver = problem_parameters['Y'].sum()
            # best case will be to always get X + initial points
            # best_case_receiver = problem_parameters['X'].sum() + math.fabs(worst_case_receiver)
            # positive_expected_utility = problem_parameters.loc[problem_parameters['E'] >= 0]['E'].sum() +\
            #                             math.fabs(worst_case_receiver)

            self.session.vars['initial_points_receiver'] = int(math.fabs(worst_case_receiver))
            # this will be calculated during the experiment,
            # and will be the sum of the non-negative payoffs in the game: 0- if Y is the result in the lottery
            # X- if X is the result --> this is the real maximum points from this experiment
            self.session.vars['max_points'] = int(math.fabs(worst_case_receiver))
            # self.session.vars['p_to_bonus_receiver'] = float(positive_expected_utility / best_case_receiver)
            # self.session.vars['best_case_receiver'] = best_case_receiver
            # self.session.vars['min_points_for_bonus_dm'] = c(int(positive_expected_utility))

            # calculate the best and worst cases for sender, and the probability to get bonus
            # num_positive_expected_utility = problem_parameters.loc[problem_parameters['E'] >= 0]['E'].count()
            # best_case_sender = Constants.num_rounds
            # self.session.vars['p_to_bonus_sender'] = float(num_positive_expected_utility / best_case_sender)
            # self.session.vars['min_points_for_bonus_ex'] = c(int(num_positive_expected_utility))

            return


class Group(BaseGroup):
    sender_answer = models.FloatField(
        verbose_name='Please provide your estimation for the probability of sampling the color red.',
                     # '\nA probability is a number between 0 and 1',
        min=0.0, max=1.0)
    receiver_choice = models.BooleanField()  # True (1) for Certainty and False (0) for Lottery
    lottery_result = models.FloatField()
    sender_timeout = models.BooleanField(choices=[True, False], initial=False)  # True if the sender had timeout
    receiver_timeout = models.BooleanField(choices=[True, False], initial=False)  # True if the receiver had timeout
    sender_payoff = models.IntegerField()
    receiver_payoff = models.IntegerField()
    x_lottery = models.FloatField()
    y_lottery = models.FloatField()
    p_lottery = models.FloatField()

    def set_payoffs(self):
        print('receiver choice:', self.receiver_choice)
        sender = self.get_player_by_role('Expert')
        receiver = self.get_player_by_role('Decision Maker')
        round_parameters = self.session.vars['problem_parameters'].loc[self.round_number-1]
        self.lottery_result = lottery(round_parameters)
        self.x_lottery = round_parameters['X']
        self.y_lottery = round_parameters['Y']
        self.p_lottery = round_parameters['P']
        if self.lottery_result > 0:  # if the lottery result is non negative - add it to max_points
            self.session.vars['max_points'] += self.lottery_result
        if self.receiver_choice:  # if the receiver choose A: Certainty - both get 0
            sender.payoff = c(0)
            receiver.payoff = c(0)
            self.sender_payoff = 0
            self.receiver_payoff = 0
        # if the receiver choose B: Lottery - sender get 1 point, and receiver get the result of the lottery
        else:
            # if there was timeout for the sender --> not get paid
            if self.sender_timeout:
                sender.payoff = c(0)
                receiver.payoff = c(self.lottery_result)
                self.sender_payoff = 0
                self.receiver_payoff = int(self.lottery_result)
            else:
                sender.payoff = c(Constants.sender_payoff_per_round)
                receiver.payoff = c(self.lottery_result)
                self.sender_payoff = Constants.sender_payoff_per_round
                self.receiver_payoff = int(self.lottery_result)

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
