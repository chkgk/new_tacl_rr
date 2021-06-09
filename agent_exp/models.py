from otree.api import (
    models, widgets, BaseConstants, BaseSubsession, BaseGroup, BasePlayer,
    Currency as c
)
import pandas as pd
import random
import os
import math


base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, 'agent_exp')
problems_data_file_path = os.path.join(data_directory, 'fix_problems_agents.csv')

author = 'Reut Apel'

doc = """The second experiment, divided into 4 different experiment with different expert's strategy"""


def accurate_zero_one(x, y, p, ev):
    return 1.0 if ev >= 0 else 0.0


def ten_accurate_zero_one(x, y, p, ev):
    return 1.0 if ev >= -0.3 else 0.0


def accurate(x, y, p, ev):
    return p


def ten_accurate(x, y, p, ev):
    return p if ev != -0.3 else (p + 0.1)


def zero_one_ly(x, y, p, ev):
    return 1.0 if (ev >= 0 or ev <= -9) else 0.0


class Constants(BaseConstants):
    name_in_url = 'agent_exp'
    players_per_group = None
    num_rounds = 50
    seconds_wait = 10
    seconds_wait_first_rounds = 30
    first_rounds = [1, 2, 3, 4, 5]
    num_rounds_long_wait = first_rounds[-1]
    bonus = 1
    # participation_fee = 2.5

    # num_timeouts_remove = 5
    minutes_to_timeout_wait_page = 10

    expert_types = {
        'accurate_zero_one': accurate_zero_one,
        'ten_accurate_zero_one': ten_accurate_zero_one,
        'accurate': accurate,
        'ten_accurate': ten_accurate,
        'zero_one_ly': zero_one_ly,
    }

    problems = pd.read_csv(problems_data_file_path, header=0)


class Subsession(BaseSubsession):

    def creating_session(self):
        """
        This function will run at the beginning of each session
        and will initial the problems parameters for all the rounds
        Each row will be the parameters of the index+1 round (index starts at 0)
        :return:
        """

        if self.round_number == 1:
            for p in self.get_players():
                if 'problem_parameters' in p.participant.vars:
                    print('already created problems')
                    continue
                problems = Constants.problems.sample(frac=1).reset_index(drop=True)
                # shuffle the problems so each subject will get them in a different order
                problems.X = problems.X.astype(float)
                problems.Y = problems.Y.astype(float)
                # Create problem set:
                p.participant.vars['problem_parameters'] = problems

                # for debug:
                print('all problems in creating_session: ', p.participant.vars['problem_parameters'],
                      'for participant', p.id_in_group)

                # calculate the best and worst cases for receiver, and the probability to get bonus
                worst_case_receiver = problems['Y'].sum()

                self.session.vars['initial_points_receiver'] = math.fabs(worst_case_receiver)
                # this will be calculated during the experiment,
                # and will be the sum of the non-negative payoffs in the game: 0- if Y is the result in the lottery
                # X- if X is the result --> this is the real maximum points from this experiment
                p.participant.vars['max_points'] = math.fabs(worst_case_receiver)

        return


class Group(BaseGroup):
    pass


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

    sender_answer = models.FloatField(min=0.0, max=1.0)
    receiver_choice = models.BooleanField()  # True (1) for Certainty and False (0) for Lottery
    sender_payoff = models.IntegerField()
    lottery_result = models.FloatField()
    receiver_timeout = models.BooleanField(choices=[True, False], initial=False)  # True if the receiver had timeout
    receiver_payoff = models.FloatField()
    x_lottery = models.FloatField()
    y_lottery = models.FloatField()
    p_lottery = models.FloatField()
    ev_lottery = models.FloatField()
    expert_type = models.StringField()

    def set_payoffs(self):
        print('receiver choice:', self.receiver_choice)
        round_parameters = self.participant.vars['problem_parameters'].loc[self.round_number - 1]
        self.lottery_result = lottery(round_parameters)
        print(f'lottery_result in set_payoff: {self.lottery_result}')
        self.x_lottery = round_parameters['X']
        self.y_lottery = round_parameters['Y']
        self.p_lottery = round_parameters['P']
        self.ev_lottery = round_parameters['E']

        if self.lottery_result > float(0):  # if the lottery result is non negative - add it to max_points
            self.participant.vars['max_points'] += self.lottery_result
        if self.receiver_choice:  # if the receiver choose A: Certainty - he get 0
            self.payoff = c(0.0)
            self.sender_payoff = 0
            self.receiver_payoff = 0.0
        # if the receiver choose B: Lottery - receiver get the result of the lottery
        else:
            self.payoff = c(self.lottery_result)
            self.sender_payoff = 1
            self.receiver_payoff = self.lottery_result

        print('receiver.payoff:', self.payoff)

        return

    def role(self):
        return {1: 'Decision Maker'}[self.id_in_group]


class Session:
    num_participants = 1
