from otree.api import Currency as c
import random
from ._builtin import Page
from .models import Constants


class Introduction(Page):
    def get_template_names(self):
        return [f"agent_exp/{self.session.config['hebrew']}Introduction.html"]

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
    template_name = 'agent_exp/PersonalInformation.html'
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
    def get_template_names(self):
        return [f"agent_exp/{self.session.config['hebrew']}ReceiverWaitPage.html"]

    timeout_seconds = 1


class ReceiverPage(Page):
    def get_template_names(self):
        return [f"agent_exp/{self.session.config['hebrew']}ReceiverPage.html"]

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
        # for debug:
        print('round_parameters for round ', self.round_number, 'are: ', round_parameters)
        # get the expert's answer according to its type
        prob_for_dm = Constants.expert_types[self.session.config['expert_type']](
            round_parameters['X'], round_parameters['Y'], round_parameters['P'], round_parameters['E'])
        self.player.sender_answer = prob_for_dm
        self.player.expert_type = self.session.config['expert_type']
        print(f'robot answer is: {self.player.sender_answer}')

        if round_parameters['Y'] == 0:
            y = 0
        else:
            y = (-1) * round_parameters['Y']
        print('prob:', prob_for_dm)
        return {
            'x': round_parameters['X'],
            'y': y,
            'p': prob_for_dm,  # the receiver will see only the sender answer for p (p') and not the real p
            'round_number': self.round_number
        }


class Results(Page):
    """This page displays the result of the round - what the receiver choose and what was the result of the lottery"""
    def get_template_names(self):
        return [f"agent_exp/{self.session.config['hebrew']}Results.html"]

    def get_timeout_seconds(self):
        if self.round_number in Constants.first_rounds:
            return 10
        else:
            return 5

    def vars_for_template(self):
        if self.player.receiver_choice:  # if the receiver chose Status quo=
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
    def get_template_names(self):
        return [f"agent_exp/{self.session.config['hebrew']}GameOver.html"]

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

        if receiver_bonus == Constants.bonus:
            player_got_bonus = True
        else:
            player_got_bonus = False

        return {'player_total_payoff': receiver_total_payoff,
                'player_bonus': self.player.participant.payoff,
                'participation_fee': self.session.config['participation_fee'],
                'player_got_bonus': player_got_bonus,
                }


page_sequence = [
    Introduction,
    PersonalInformation,
    ReceiverWaitPage,
    ReceiverPage,
    Results,
    GameOver,
]
