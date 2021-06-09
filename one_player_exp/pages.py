from otree.api import Currency as c
import random
from ._builtin import Page
from .models import Constants


class Introduction(Page):
    template_name = 'one_player_exp/Introduction.html'

    def is_displayed(self):
        return self.round_number == 1  # show this page only in round 1: in the beginning of the game

    def vars_for_template(self):
        self.subsession.creating_session()
        return {
            'participation_fee': self.session.config['participation_fee'],
            'initial_points_dc': self.session.vars['initial_points_receiver'],
            'expert_type': self.session.config['expert_type'],
        }


class PersonalInformation(Page):
    template_name = 'one_player_exp/PersonalInformation.html'
    form_model = 'player'
    form_fields = ['name', 'age', 'gender', 'is_student', 'occupation', 'residence']

    def is_displayed(self):
        return self.round_number == 1  # show this page only in round 1: in the beginning of the game

    def before_next_page(self):
        self.participant.vars['name'] = self.player.name
        self.participant.vars['age'] = self.player.age
        self.participant.vars['gender'] = self.player.gender
        self.participant.vars['occupation'] = self.player.occupation
        self.participant.vars['is_student'] = self.player.is_student
        self.participant.vars['residence'] = self.player.residence
        self.participant.vars['num_timeout'] = 0


class ReceiverWaitPage(Page):
    template_name = 'one_player_exp/ReceiverWaitPage.html'

    timeout_seconds = 1


class ReceiverPage(Page):
    template_name = 'one_player_exp/ReceiverPage.html'

    form_model = 'player'
    form_fields = ['receiver_choice']

    timeout_submission = {'receiver_choice': bool(random.getrandbits(1))}
    # if the receiver didn't choose his choice after seconds_wait, an option will be chosen randomly

    def get_timeout_seconds(self):
        if self.round_number in Constants.first_rounds:
            return Constants.seconds_wait_first_rounds
        else:
            return Constants.seconds_wait

    def before_next_page(self):
        if self.timeout_happened:  # if the receiver didn't provide a probability after seconds_wait
            self.player.receiver_timeout = True
            print('receiver timeout:', self.player.receiver_timeout)

        self.player.set_payoffs()  # run this only once - after the receiver choice

    def vars_for_template(self):
        round_parameters = self.player.participant.vars['problem_parameters'].loc[self.round_number-1]
        self.player.sender_answer = round_parameters['P']
        self.player.expert_type = self.session.config['expert_type']

        if round_parameters['Y'] == 0:
            y = 0
        else:
            y = (-1) * round_parameters['Y']
        return {
            'x': round_parameters['X'],
            'y': y,
            'p': round_parameters['P'],
            '1_p': round(1 - round_parameters['P'], 1),
            'round_number': self.round_number
        }


class Results(Page):
    """This page displays the result of the round - what the receiver choose and what was the result of the lottery"""
    template_name = 'one_player_exp/Results.html'

    def get_timeout_seconds(self):
        if self.round_number in Constants.first_rounds:
            return 10
        else:
            return 5

    def vars_for_template(self):
        if self.player.receiver_choice:  # if the receiver chose Status quo
            other_choice = 'Action'
            receiver_choice = 'Status quo'

            other_gain_receiver = abs(self.player.lottery_result)
            receiver_payoff = 0
            if self.player.lottery_result < 0:
                negative_other_gain_receiver = True
            else:
                negative_other_gain_receiver = False
        else:
            receiver_choice = 'Action'
            other_choice = 'Status quo'
            other_gain_receiver = 0
            receiver_payoff = self.player.lottery_result
            negative_other_gain_receiver = False

        print('lottery_result:', self.player.lottery_result)
        print('receiver_payoff:', receiver_payoff)

        return {
            'round': self.round_number,
            'receiver_choice': receiver_choice,
            'other_choice': other_choice,
            'lottery_result': self.player.lottery_result,
            'other_gain_receiver': other_gain_receiver,
            'receiver_payoff': receiver_payoff,
            'receiver_negative_result': abs(self.player.lottery_result),
            'negative_other_gain_receiver': negative_other_gain_receiver,
            'receiver_timeout': self.player.receiver_timeout
        }


class GameOver(Page):
    """
    This page will be displayed after the last round is over - the experiment is finish.
    It will display the results: the payoff of each player
    """
    template_name = 'one_player_exp/GameOver.html'

    def is_displayed(self):
        # show this page only after the last round
        return self.round_number == Constants.num_rounds

    def vars_for_template(self):
        # get the number of points of each player and convert to real money
        # receiver total points
        receiver_total_points = sum([p.payoff for p in self.player.in_all_rounds()]) + \
                                self.session.vars['initial_points_receiver']
        print('receiver_total_points', receiver_total_points)
        receiver_p_to_bonus = float(receiver_total_points / self.player.participant.vars['max_points'])
        print('receiver_p_to_bonus:', receiver_p_to_bonus)
        receiver_bonus = Constants.bonus if random.random() <= receiver_p_to_bonus else 0
        print('receiver_bonus', receiver_bonus)
        receiver_total_payoff = receiver_bonus + self.session.config['participation_fee']
        receiver_total_payoff = c(receiver_total_payoff)
        print('receiver payoff', receiver_total_payoff)

        print('max points for receiver:', self.player.participant.vars['max_points'])

        self.player.participant.payoff = c(receiver_bonus)
        print('receiver final payoff:', self.player.participant.payoff)

        return {'player_total_payoff': receiver_total_payoff,
                'player_bonus': self.player.participant.payoff,
                'participation_fee': self.session.config['participation_fee'],
                }


page_sequence = [
    Introduction,
    PersonalInformation,
    ReceiverPage,
    Results,
    GameOver,
]
