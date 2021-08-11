from otree.api import (
    models, widgets, BaseConstants, BaseSubsession, BaseGroup, BasePlayer,
    Currency as c
)
import pandas as pd
import random
import os
import math
import time
#import pandas as pd
from background.tasks import huey

base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, 'text_exp_bot_bg')
reviews_not_seen_file_path = os.path.join(data_directory, 'reviews_not_seen.csv')
image_directory = os.path.join(base_directory, '_static', 'text_exp_bot_bg')

author = 'Maya Tarno'

doc = """
"""


class Constants(BaseConstants):
    name_in_url = 'text_exp_bot_bg'
    players_per_group = None
    num_rounds = 10
    sender_payoff_per_round = 1
    seconds_wait_expert = 240  # 120
    seconds_wait_first_rounds_expert = 240  # 150
    seconds_wait_dm = 240
    seconds_wait_first_rounds_dm = 240
    first_rounds = [1]
    num_rounds_long_wait = first_rounds[-1]
    bonus = 1
    number_reviews_per_hotel = 7
    cost = 8
    rounds_for_reviews_in_test = [7, 9]
    num_timeouts_remove = 5
    minutes_to_timeout_wait_page = 10
    pay_by_time_for_waiting = 0.04
    max_payment_for_waiting = 0.4
    reviews_not_seen = pd.read_csv(reviews_not_seen_file_path, header=0)
    intro_timeout_minutes = 10
    real_participation_fee = 2.5


class Subsession(BaseSubsession):
    # ('class Subsession')
    condition = models.StringField()

    def creating_session(self):
        # print('creating_session!!!')
        """
        This function will run at the beginning of each session
        and will initial the problems parameters for all the rounds
        Each row will be the parameters of the index+1 round (index starts at 0)
        :return:
        """

        problems_data_file_path = os.path.join(data_directory, f"10_reviews_test_data.csv")
        problems = pd.read_csv(problems_data_file_path, header=0).sample(frac=1).reset_index(drop=True)
        if self.round_number == 1:
            for g in self.get_groups():
                if g.set_parameters:
                    continue

                for p in g.get_players():
                    if p.id_in_group == 1:  # create the parameters only for experts:
                        # Load problems and shuffle the them so each subject will get them in a different order
                        # problems = Constants.problems
                        for i in range(Constants.number_reviews_per_hotel):
                            problems[f'score_{i}'] = problems[f'score_{i}'].astype(float)
                            # insert the positive and then the negative or the negative and then the positive randomly
                            for hotel_num in range(Constants.num_rounds):
                                if bool(random.getrandbits(1)):
                                    problems.loc[hotel_num, f'random_positive_negative_review_{i}'] = \
                                        problems.loc[hotel_num, f'negative_review_{i}'] + ' ' + problems.loc[
                                            hotel_num, f'positive_review_{i}']
                                else:
                                    problems.loc[hotel_num, f'random_positive_negative_review_{i}'] = \
                                        problems.loc[hotel_num, f'positive_review_{i}'] + ' ' + problems.loc[
                                            hotel_num, f'negative_review_{i}']
                        # Create problem set:
                        p.participant.vars['problem_parameters'] = problems
                        round_parameters = pd.DataFrame(
                            columns=['pair_id', 'review_id', 'subsession_round_number', 'lottery_result_low',
                                     'lottery_result_med1', 'lottery_result_high', 'group_lottery_result',
                                     'group_receiver_payoff', 'group_sender_answer_index', 'group_sender_payoff',
                                     'group_sender_answer_reviews', 'group_average_score' 'chose_lose', 'chose_earn',
                                     'not_chose_lose', 'not_chose_earn', 'group_sender_answer_scores', 'group_score_0',
                                     'group_score_1', 'group_score_2', 'group_score_3', 'group_score_4',
                                     'group_score_5', 'group_score_6', 'group_random_positive_negative_review_0',
                                     'group_random_positive_negative_review_1',
                                     'group_random_positive_negative_review_2',
                                     'group_random_positive_negative_review_3',
                                     'group_random_positive_negative_review_4',
                                     'group_random_positive_negative_review_5',
                                     'group_random_positive_negative_review_6', 'group_positive_review_5',
                                     'group_positive_review_6', 'group_positive_review_4', 'group_positive_review_3',
                                     'group_positive_review_2', 'group_positive_review_1', 'group_positive_review_0',
                                     'group_negative_review_0', 'group_negative_review_1', 'group_negative_review_2',
                                     'group_negative_review_3', 'group_negative_review_4', 'group_negative_review_5',
                                     'group_negative_review_6', 'status', 'player_id_in_group'])
                        for round_ in range(1, 11):
                            # print(self.participant.vars['problem_parameters']['average_score'],'lplp')
                            round_parameters.at[round_ - 1, 'player_id_in_group'] = 2
                            round_parameters.at[round_ - 1, 'status'] = 'play'
                            round_parameters.at[round_ - 1, 'group_negative_review_0'] =\
                                problems.at[round_ - 1, 'negative_review_0']
                            round_parameters.at[round_ - 1, 'group_negative_review_1'] = \
                                problems.at[round_ - 1, 'negative_review_1']
                            round_parameters.at[round_ - 1, 'group_negative_review_2'] = \
                                problems.at[round_ - 1, 'negative_review_2']
                            round_parameters.at[round_ - 1, 'group_negative_review_3'] = \
                                problems.at[round_ - 1, 'negative_review_3']
                            round_parameters.at[round_ - 1, 'group_negative_review_4'] = \
                                problems.at[round_ - 1, 'negative_review_4']
                            round_parameters.at[round_ - 1, 'group_negative_review_5'] = \
                                problems.at[round_ - 1, 'negative_review_5']
                            round_parameters.at[round_ - 1, 'group_negative_review_6'] = \
                                problems.at[round_ - 1, 'negative_review_6']
                            round_parameters.at[round_ - 1, 'group_positive_review_5'] = \
                                problems.at[round_ - 1, 'positive_review_5']
                            round_parameters.at[round_ - 1, 'group_positive_review_4'] = \
                                problems.at[round_ - 1, 'positive_review_4']
                            round_parameters.at[round_ - 1, 'group_positive_review_3'] = \
                                problems.at[round_ - 1, 'positive_review_3']
                            round_parameters.at[round_ - 1, 'group_positive_review_1'] = \
                                problems.at[round_ - 1, 'positive_review_1']
                            round_parameters.at[round_ - 1, 'group_positive_review_2'] = \
                                problems.at[round_ - 1, 'positive_review_2']
                            round_parameters.at[round_ - 1, 'group_positive_review_0'] = \
                                problems.at[round_ - 1, 'positive_review_0']
                            round_parameters.at[round_ - 1, 'group_random_positive_negative_review_0'] = \
                                problems.at[round_ - 1, 'random_positive_negative_review_0']
                            round_parameters.at[round_ - 1, 'group_random_positive_negative_review_1'] = \
                                problems.at[round_ - 1, 'random_positive_negative_review_1']
                            round_parameters.at[round_ - 1, 'group_random_positive_negative_review_2'] = \
                                problems.at[round_ - 1, 'random_positive_negative_review_2']
                            round_parameters.at[round_ - 1, 'group_random_positive_negative_review_3'] = \
                                problems.at[round_ - 1, 'random_positive_negative_review_3']
                            round_parameters.at[round_ - 1, 'group_random_positive_negative_review_4'] = \
                                problems.at[round_ - 1, 'random_positive_negative_review_4']
                            round_parameters.at[round_ - 1, 'group_random_positive_negative_review_5'] = \
                                problems.at[round_ - 1, 'random_positive_negative_review_5']
                            round_parameters.at[round_ - 1, 'group_random_positive_negative_review_6'] = \
                                problems.at[round_ - 1, 'random_positive_negative_review_6']
                            round_parameters.at[round_ - 1, 'group_score_0'] = problems.at[round_ - 1, 'score_0']
                            round_parameters.at[round_ - 1, 'group_score_1'] = problems.at[round_ - 1, 'score_1']
                            round_parameters.at[round_ - 1, 'group_score_2'] = problems.at[round_ - 1, 'score_2']
                            round_parameters.at[round_ - 1, 'group_score_3'] = problems.at[round_ - 1, 'score_3']
                            round_parameters.at[round_ - 1, 'group_score_4'] = problems.at[round_ - 1, 'score_4']
                            round_parameters.at[round_ - 1, 'group_score_5'] = problems.at[round_ - 1, 'score_5']
                            round_parameters.at[round_ - 1, 'group_score_6'] = problems.at[round_ - 1, 'score_6']
                            round_parameters.at[round_ - 1, 'subsession_round_number'] = round_
                            round_parameters.at[round_ - 1, 'group_average_score'] =\
                                round(problems.at[round_ - 1, 'average_score'], 2)

                        p.participant.vars['round_parameters'] = round_parameters
                        g.set_parameters = True

                        # for debug:
                        # print('all problems in creating_session: ', p.participant.vars['problem_parameters'])

                        # calculate the worst cases for receiver, and the probability to get bonus
                        worst_case_receiver = round((problems['min_score']-Constants.cost).sum(), 2)
                        self.session.vars['initial_points_receiver'] = math.fabs(worst_case_receiver)
                        # this will be calculated during the experiment, and will be the sum of the non-negative
                        # payoffs in the game: 0- if Y is the result in the lottery
                        # X- if X is the result --> this is the real maximum points from this experiment
                        p.participant.vars['max_points'] = math.fabs(worst_case_receiver)

        return


class Group(BaseGroup):
    # print('Group class defining')
    set_parameters = models.BooleanField(initial=False)
    receiver_choice = models.BooleanField()  # True (1) for Certainty and False (0) for Lottery
    lottery_result = models.FloatField()
    sender_timeout = models.BooleanField(choices=[True, False], initial=False)  # True if the sender had timeout
    receiver_timeout = models.BooleanField(choices=[True, False], initial=False)  # True if the receiver had timeout
    sender_payoff = models.IntegerField()
    receiver_payoff = models.FloatField()
    receiver_passed_test = models.FloatField()
    failed_intro_test = models.BooleanField(choices=[True, False], initial=False)
    instruction_timeout = models.BooleanField(choices=[True, False], initial=False)
    pass_intro_test = models.BooleanField()
    # reviews and scores
    is_done = models.BooleanField()
    score_0 = models.FloatField()
    score_1 = models.FloatField()
    score_2 = models.FloatField()
    score_3 = models.FloatField()
    score_4 = models.FloatField()
    score_5 = models.FloatField()
    score_6 = models.FloatField()
    review_0 = models.LongStringField()
    review_1 = models.LongStringField()
    review_2 = models.LongStringField()
    review_3 = models.LongStringField()
    review_4 = models.LongStringField()
    review_5 = models.LongStringField()
    review_6 = models.LongStringField()
    positive_review_0 = models.LongStringField()
    positive_review_1 = models.LongStringField()
    positive_review_2 = models.LongStringField()
    positive_review_3 = models.LongStringField()
    positive_review_4 = models.LongStringField()
    positive_review_5 = models.LongStringField()
    positive_review_6 = models.LongStringField()
    negative_review_0 = models.LongStringField()
    negative_review_1 = models.LongStringField()
    negative_review_2 = models.LongStringField()
    negative_review_3 = models.LongStringField()
    negative_review_4 = models.LongStringField()
    negative_review_5 = models.LongStringField()
    negative_review_6 = models.LongStringField()
    random_positive_negative_review_0 = models.LongStringField()
    random_positive_negative_review_1 = models.LongStringField()
    random_positive_negative_review_2 = models.LongStringField()
    random_positive_negative_review_3 = models.LongStringField()
    random_positive_negative_review_4 = models.LongStringField()
    random_positive_negative_review_5 = models.LongStringField()
    random_positive_negative_review_6 = models.LongStringField()
    average_score = models.FloatField()

    sender_answer_reviews = models.LongStringField()
    sender_answer_negative_reviews = models.LongStringField()
    sender_answer_positive_reviews = models.LongStringField()
    sender_answer_scores = models.FloatField()
    sender_answer_index = models.IntegerField(widget=widgets.RadioSelect,
                                              choices=list(range(1, Constants.number_reviews_per_hotel+1)))

    finish_mcts = models.BooleanField(initial=False)
    receiver_finish_round = models.BooleanField(initial=False)
    action = models.IntegerField(initial=10)

    def set_round_parameters(self):
        start_time = time.time()
        #print('set_round_parameters group')
        sender = self.get_players()[0]
        #print(sender,'sender')
        # create lottery result and insert round_parameters to the database
        #for round_number in range(1,11):
        #    print(f"hotel_for_round_{round_number - 1}_is_{sender.participant.vars['problem_parameters'].loc[round_number - 1]['average_score']}")
        round_parameters = sender.participant.vars['problem_parameters'].loc[self.round_number - 1]
        self.average_score = round_parameters['average_score']
        self.is_done = False
        self.score_0 = round_parameters['score_0']
        self.score_1 = round_parameters['score_1']
        self.score_2 = round_parameters['score_2']
        self.score_3 = round_parameters['score_3']
        self.score_4 = round_parameters['score_4']
        self.score_5 = round_parameters['score_5']
        self.score_6 = round_parameters['score_6']
        self.review_0 = round_parameters['review_0']
        self.review_1 = round_parameters['review_1']
        self.review_2 = round_parameters['review_2']
        self.review_3 = round_parameters['review_3']
        self.review_4 = round_parameters['review_4']
        self.review_5 = round_parameters['review_5']
        self.review_6 = round_parameters['review_6']
        self.positive_review_0 = round_parameters['positive_review_0']
        self.positive_review_1 = round_parameters['positive_review_1']
        self.positive_review_2 = round_parameters['positive_review_2']
        self.positive_review_3 = round_parameters['positive_review_3']
        self.positive_review_4 = round_parameters['positive_review_4']
        self.positive_review_5 = round_parameters['positive_review_5']
        self.positive_review_6 = round_parameters['positive_review_6']
        self.negative_review_0 = round_parameters['negative_review_0']
        self.negative_review_1 = round_parameters['negative_review_1']
        self.negative_review_2 = round_parameters['negative_review_2']
        self.negative_review_3 = round_parameters['negative_review_3']
        self.negative_review_4 = round_parameters['negative_review_4']
        self.negative_review_5 = round_parameters['negative_review_5']
        self.negative_review_6 = round_parameters['negative_review_6']
        self.random_positive_negative_review_0 = round_parameters['random_positive_negative_review_0']
        self.random_positive_negative_review_1 = round_parameters['random_positive_negative_review_1']
        self.random_positive_negative_review_2 = round_parameters['random_positive_negative_review_2']
        self.random_positive_negative_review_3 = round_parameters['random_positive_negative_review_3']
        self.random_positive_negative_review_4 = round_parameters['random_positive_negative_review_4']
        self.random_positive_negative_review_5 = round_parameters['random_positive_negative_review_5']
        self.random_positive_negative_review_6 = round_parameters['random_positive_negative_review_6']

        score_columns = [f'score_{i}' for i in range(Constants.number_reviews_per_hotel)]
        score_list = round_parameters[score_columns]
        # print(f'score_list: {score_list}')
        self.lottery_result = lottery(score_list)
        # (f'lottery result:{self.lottery_result}')
        # print("--- group class seconds ---" % (time.time() - start_time))

        # print(f'scores parameters in set_round_parameters: '
        #       f'{self.score_0}, {self.score_1}, {self.score_2}, {self.score_3}, {self.score_4}, {self.score_5}'
        #       f', {self.score_6}')

        # if the lottery result is higher than the cost - add it to max_points
        if self.lottery_result > Constants.cost:
            sender.participant.vars['max_points'] += (self.lottery_result - Constants.cost)

    def set_payoffs(self):
        # print('set_payoffs group')
        # print('receiver choice:', self.receiver_choice, "maya: ", self.get_player_by_role('Decision Maker'))
        # sender = self.get_player_by_role('Expert')
        receiver = self.get_player_by_role('Decision Maker')

        if self.receiver_choice:  # if the receiver choose A: Certainty - both get 0
            # sender.payoff = c(0)
            receiver.payoff = c(0.0)
            self.receiver_payoff = 0.0
            self.sender_payoff = 0
            # self.receiver_payoff = float(0)
            # print('self.receiver_choice:', self.receiver_choice, 'self.receiver_payoff:', self.receiver_payoff,
            #       'self.lottery_result:', self.lottery_result)

        # if the receiver choose B: Lottery - sender get 1 point, and receiver get the result of the lottery
        else:
            receiver.payoff = c(round(self.lottery_result - Constants.cost, 1))
            self.receiver_payoff = round(self.lottery_result - Constants.cost, 1)
            # self.receiver_payoff = self.lottery_result
            # print('self.receiver_choice:', self.receiver_choice, 'self.receiver_payoff:', self.receiver_payoff,
            #       'self.lottery_result:', self.lottery_result)

            # if there was timeout for the sender --> not get paid

            # sender.payoff = c(Constants.sender_payoff_per_round)
            self.sender_payoff = Constants.sender_payoff_per_round

        # print('receiver.payoff:', receiver.payoff, 'sender.payoff:', self.sender_payoff)

        return

    # def is_player_drop_out(self):
    #     """
    #     :return: This function return True if there are less then 2 players in the group - one player drop out,
    #     and False otherwise
    #     """
    #     players_with_us = self.get_players()
    #     return True if len(players_with_us) != Constants.players_per_group else False


def lottery(score_list):
    return random.choice(score_list)


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

    intro_test = models.StringField(
        verbose_name='Do you have any comments on this HIT?',
        initial=''
    )

    # DM test chosen and not chosen reviews
    dm_test_chosen_review_1 = models.BooleanField(
        verbose_name='Did you see this during the experiment?',
        widget=widgets.RadioSelect
    )
    dm_test_chosen_review_2 = models.BooleanField(
        verbose_name='Did you see this during the experiment?',
        widget=widgets.RadioSelect
    )

    dm_test_not_chosen_review_1 = models.BooleanField(
        verbose_name='Did you see this during the experiment?',
        widget=widgets.RadioSelect
    )
    dm_test_not_chosen_review_2 = models.BooleanField(
        verbose_name='Did you see this during the experiment?',
        widget=widgets.RadioSelect
    )
    action = models.IntegerField()
    action_id = models.StringField(default='')


    def role(self):
        return {1: 'Decision Maker'}[self.id_in_group]#1: 'Expert',

    # def is_displayed(self):
    #     return self.participant.vars['num_timeout'] < 5

    def live_resultcheck(self, data):
        if data['message'] == 'get_result':
            try:
                print('try to get result with id', self.action_id)
                action = huey.result(self.action_id)
            except TypeError:
                action = None
            if action:
                print('got the result!')
                self.action = action
                # self.group.sender_answer_index = self.group.action if self.group.action is not None else 6
                # round_parameters = self.participant.vars['problem_parameters'].loc[self.group.round_number - 1]
                # self.group.sender_answer_scores = round_parameters[f'score_{self.group.sender_answer_index}']
                # self.group.sender_answer_reviews = round_parameters[f'random_positive_negative_review_{self.group.sender_answer_index}']
                # self.group.sender_answer_positive_reviews = round_parameters[f'positive_review_{self.group.sender_answer_index}']
                # self.group.sender_answer_negative_reviews = round_parameters[f'negative_review_{self.group.sender_answer_index}']
                return {self.id_in_group: {'message': 'calculation_done'}}


class Session:
    #print('Session class')
    num_participants = 1
    #problems_data_file_path = os.path.join(data_directory, f"{self.session.config['review_file_name']}.csv")
    #problems = pd.read_csv(problems_data_file_path, header=0).sample(frac=1).reset_index(drop=True)

