from otree.api import (
    models, widgets, BaseConstants, BaseSubsession, BaseGroup, BasePlayer,
    Currency as c
)
import pandas as pd
import random
import os
import math
import copy

base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, 'text_exp_deterministic')
problems_data_file_path = os.path.join(data_directory, '10_reviews.csv')
reviews_not_seen_file_path = os.path.join(data_directory, 'reviews_not_seen.csv')
image_directory = os.path.join(base_directory, '_static', 'text_exp_deterministic')

author = 'Reut Apel'

doc = """
"""


class Constants(BaseConstants):
    name_in_url = 'text_exp_deterministic'
    players_per_group = 2
    num_rounds = 10
    sender_payoff_per_round = 1
    seconds_wait_expert = 120
    seconds_wait_first_rounds_expert = 150
    seconds_wait_dm = 40
    seconds_wait_first_rounds_dm = 60
    first_rounds = [1, 2]
    num_rounds_long_wait = first_rounds[-1]
    bonus = 1
    number_reviews_per_hotel = 7
    cost = 8
    rounds_for_reviews_in_test = [7, 9]
    # participation_fee = 2.5

    num_timeouts_remove = 5
    minutes_to_timeout_wait_page = 10
    pay_by_time_for_waiting = 0.04
    max_payment_for_waiting = 0.4

    problems = pd.read_csv(problems_data_file_path, header=0).sample(frac=1).reset_index(drop=True)
    reviews_not_seen = pd.read_csv(reviews_not_seen_file_path, header=0)
    intro_timeout_minutes = 10
    real_participation_fee = 2.5


class Subsession(BaseSubsession):
    condition = models.StringField()

    def creating_session(self):
        """
        This function will run at the beginning of each session
        and will initial the problems parameters for all the rounds
        Each row will be the parameters of the index+1 round (index starts at 0)
        :return:
        """
        self.condition = self.session.config['cond']
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
                        print(f'set parameters to True')
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
    chosen_index = models.IntegerField(choices=list(range(1, Constants.number_reviews_per_hotel+1)))

    def set_round_parameters(self):
        sender = self.get_player_by_role('Expert')
        # create lottery result and insert round_parameters to the database
        round_parameters = sender.participant.vars['problem_parameters'].loc[self.round_number - 1]

        self.average_score = round_parameters['average_score']
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

        if f'lottery_result_round_{self.round_number}' not in sender.participant.vars:
            # choose index and not score
            self.chosen_index = copy.deepcopy(lottery(list(range(1, Constants.number_reviews_per_hotel+1))))
            self.lottery_result = round_parameters[f'score_{self.chosen_index-1}']
            sender.participant.vars[f'lottery_result_round_{self.round_number}'] = copy.deepcopy(self.lottery_result)
            # if the lottery result is higher than the cost - add it to max_points
            if sender.participant.vars[f'lottery_result_round_{self.round_number}'] > Constants.cost:
                sender.participant.vars['max_points'] += \
                    (sender.participant.vars[f'lottery_result_round_{self.round_number}'] - Constants.cost)
        print(f'in set_round_parameters: the lottery result is: {self.lottery_result}')
        print(f'in set_round_parameters: the lottery result in round {self.round_number} is: '
              f'{sender.participant.vars[f"lottery_result_round_{self.round_number}"]}')

        # print(f'scores parameters in set_round_parameters: '
        #       f'{self.score_0}, {self.score_1}, {self.score_2}, {self.score_3}, {self.score_4}, {self.score_5}'
        #       f', {self.score_6}')

    def set_payoffs(self):
        print('receiver choice:', self.receiver_choice)
        sender = self.get_player_by_role('Expert')
        receiver = self.get_player_by_role('Decision Maker')
        print(f'in set_payoffs: the lottery result in round {self.round_number} is: '
              f'{sender.participant.vars[f"lottery_result_round_{self.round_number}"]}')

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
            receiver.payoff =\
                c(round(sender.participant.vars[f'lottery_result_round_{self.round_number}'] - Constants.cost, 1))
            self.receiver_payoff =\
                round(sender.participant.vars[f'lottery_result_round_{self.round_number}'] - Constants.cost, 1)
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
    # dm_test_chosen_review_3 = models.BooleanField(
    #     verbose_name='Did you see this text during the experiment?',
    #     widget=widgets.RadioSelect
    # )
    dm_test_not_chosen_review_1 = models.BooleanField(
        verbose_name='Did you see this during the experiment?',
        widget=widgets.RadioSelect
    )
    dm_test_not_chosen_review_2 = models.BooleanField(
        verbose_name='Did you see this during the experiment?',
        widget=widgets.RadioSelect
    )
    # dm_test_not_chosen_review_3 = models.BooleanField(
    #     verbose_name='Did you see this text during the experiment?',
    #     widget=widgets.RadioSelect
    # )

    def role(self):
        return {1: 'Expert', 2: 'Decision Maker'}[self.id_in_group]

    # def is_displayed(self):
    #     return self.participant.vars['num_timeout'] < 5


class Session:
    num_participants = 2
