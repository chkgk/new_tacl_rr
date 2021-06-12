from otree.api import (
    models, widgets, BaseConstants, BaseSubsession, BaseGroup, BasePlayer,
    Currency as c
)
import pandas as pd
import random
import os
import math


base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, 'mturk_ag_second_exp')
problems_data_file_path = os.path.join(data_directory, '2_gambles_exp.csv')

author = 'Reut Apel'

doc = """The second experiment with agent, divided into 2 different experiment with different expert's strategy,
different problems are checked"""


def zero_one(corr, p_x_u, ev_u, p_win):
    return 1.0 if corr == 1 else 0.0


def probability(corr, p_x_u, ev_u, p_win):
    return round(p_x_u + 0.1, 1) if ev_u == -1 else round(p_x_u - 0.1, 1)


def p_win_agent(corr, p_x_u, ev_u, p_win):
    return p_win


class Constants(BaseConstants):
    name_in_url = 'mturk_ag_second_exp'
    players_per_group = None
    num_rounds = 60
    seconds_wait = 20
    seconds_wait_first_rounds = 30
    first_rounds = [1, 2, 3, 4, 5]
    num_rounds_long_wait = first_rounds[-1]
    bonus = 1
    # participation_fee = 2.5

    # num_timeouts_remove = 5
    minutes_to_timeout_wait_page = 10

    expert_types = {
        'zero_one': zero_one,
        'probability': probability,
        'p_win_agent': p_win_agent,
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
                # shuffle the problems so each subject will get them in a different order
                problems = Constants.problems.sample(frac=1).reset_index(drop=True)

                # Create problem set:
                p.participant.vars['problem_parameters'] = problems

                # for debug:
                print('all problems in creating_session: ', p.participant.vars['problem_parameters'],
                      'for participant', p.id_in_group)

                # calculate the worst case for receiver, and the probability to get bonus. y_k<y_u for all problems so
                # the worst case is when always chose the known and gor negative
                worst_case_receiver = problems['y_k'].sum()

                self.session.vars['initial_points_receiver'] = math.fabs(worst_case_receiver)
                # this will be calculated during the experiment,
                # and will be the sum of the non-negative payoffs in the game: 0- if Y is the result in the lottery
                # X- if X is the result --> this is the real maximum points from this experiment
                p.participant.vars['max_points'] = math.fabs(worst_case_receiver)

        return


class Group(BaseGroup):
    pass


def lottery(round_parameters):
    # get a random number in [0,1].
    chance = random.random()
    if chance <= round_parameters['p_x_k']:
        k_lottery_result = round_parameters['x_k']
    elif chance <= round_parameters['p_y_k'] + round_parameters['p_x_k']:
        k_lottery_result = round_parameters['y_k']
    else:
        k_lottery_result = round_parameters['z_k']

    if round_parameters['corr'] == 1:
        u_lottery_result = round_parameters['x_u'] if chance <= round_parameters['p_x_u'] else round_parameters['y_u']
    else:
        u_lottery_result = round_parameters['x_u'] if (1-chance) <= round_parameters['p_x_u']\
            else round_parameters['y_u']

    return k_lottery_result, u_lottery_result


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
    receiver_choice = models.BooleanField()  # True (1) for Known and False (0) for Unknown
    sender_payoff = models.IntegerField()
    k_lottery_result = models.IntegerField()
    u_lottery_result = models.IntegerField()
    receiver_timeout = models.BooleanField(choices=[True, False], initial=False)  # True if the receiver had timeout
    receiver_payoff = models.FloatField()
    # the known option parameters
    x_k_lottery = models.IntegerField()
    y_k_lottery = models.IntegerField()
    z_k_lottery = models.IntegerField()
    p_x_k_lottery = models.FloatField()
    p_y_k_lottery = models.FloatField()
    p_z_k_lottery = models.FloatField()

    # the unknown option parameters
    x_u_lottery = models.IntegerField()
    y_u_lottery = models.IntegerField()
    p_x_u_lottery = models.FloatField()
    p_y_u_lottery = models.FloatField()
    ev_u_lottery = models.FloatField()

    expert_type = models.StringField()
    corr = models.IntegerField(choices=(1, -1))

    def set_payoffs(self):
        print('receiver choice:', self.receiver_choice)
        round_parameters = self.participant.vars['problem_parameters'].loc[self.round_number - 1]
        self.k_lottery_result, self.u_lottery_result = lottery(round_parameters)
        print(f'k_lottery_result in set_payoff: {self.k_lottery_result} u_lottery_result: {self.u_lottery_result}')
        # the known option parameters
        self.x_k_lottery = round_parameters['x_k']
        self.y_k_lottery = round_parameters['y_k']
        self.z_k_lottery = round_parameters['z_k']
        self.p_x_k_lottery = round_parameters['p_x_k']
        self.p_y_k_lottery = round_parameters['p_y_k']
        self.p_z_k_lottery = round_parameters['p_z_k']

        # the unknown option parameters
        self.x_u_lottery = round_parameters['x_u']
        self.y_u_lottery = round_parameters['y_u']
        self.p_x_u_lottery = round_parameters['p_x_u']
        self.p_y_u_lottery = round_parameters['p_y_u']

        self.ev_u_lottery = round_parameters['ev_u']
        self.corr = round_parameters['corr']

        # if the lottery result is non negative - add the max between the lottery_results to max_points
        if self.u_lottery_result > 0 or self.k_lottery_result > 0:
            self.participant.vars['max_points'] += max(self.u_lottery_result, self.k_lottery_result)
        if self.receiver_choice:  # if the receiver choose Known - he gets k_lottery_result
            self.payoff = c(self.k_lottery_result)
            self.sender_payoff = 0
            self.receiver_payoff = self.k_lottery_result
        # if the receiver choose Unknown - receiver gets u_lottery_result
        else:
            self.payoff = c(self.u_lottery_result)
            self.sender_payoff = 1
            self.receiver_payoff = self.u_lottery_result

        print('receiver.payoff:', self.payoff)

        return

    def role(self):
        return {1: 'Decision Maker'}[self.id_in_group]


class Session:
    num_participants = 1
