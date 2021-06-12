from otree.api import Currency as c, currency_range
import random
from otree_mturk_utils.pages import CustomMturkPage, CustomMturkWaitPage
from ._builtin import Page, WaitPage
from .models import Constants


class GroupedWaitPage(CustomMturkWaitPage):
    group_by_arrival_time = True
    pay_by_time = Constants.pay_by_time_for_waiting / 60  # they will get 12 cents for a minute they wait
    startwp_timer = 360  # after 6 minutes they can skip to the end of the experiment
    startwp_timer_to_timeout = 600  # after 10 minutes the experiment will automatically end
    use_task = False

    def is_displayed(self):
        return self.round_number == 1


class Introduction(CustomMturkPage):
    template_name = 'first_fix_prob_exp/Introduction.html'

    def is_displayed(self):
        return self.round_number == 1  # show this page only in round 1: in the beginning of the game

    def vars_for_template(self):
        if self.player.id_in_group == 1:  # run this once
            self.subsession.creating_session()
        if self.player.id_in_group == 1:
            other_role = 'Decision Maker'
        else:
            other_role = 'Expert'

        return {
            'participation_fee': self.session.config['participation_fee'],
            'other_role': other_role,
            'initial_points_dc': self.session.vars['initial_points_receiver'],
        }


class PersonalInformation(CustomMturkPage):
    template_name = 'first_fix_prob_exp/PersonalInformation.html'
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


class AfterInstructions(WaitPage):
    def is_displayed(self):
        return self.round_number == 1  # show this page only in round 1: in the beginning of the game


class SenderPage(CustomMturkPage):
    template_name = 'first_fix_prob_exp/SenderPage.html'
    form_model = 'group'
    form_fields = ['sender_answer']

    timeout_submission = {'sender_timeout': True}
    # if the sender didn't provide a probability after seconds_wait,
    # the probability we will show to the receiver will be ?

    def get_timeout_seconds(self):
        if self.round_number in Constants.first_rounds:
            return Constants.seconds_wait_first_rounds
        else:
            return Constants.seconds_wait

    def before_next_page(self):
        if self.timeout_happened:  # if the sender didn't provide a probability after seconds_wait
            # self.participant.vars['num_timeout'] += 1
            self.group.sender_timeout = True
            print('sender timeout:', self.group.sender_timeout)
            # if self.participant.vars['num_timeout'] > Constants.num_timeouts_remove:
            #     self.session.vars['remove_player'] = True
            #     self.session.vars['player_to_remove'] = 'Expert'
            #     self.session.vars['player_not_remove'] = 'Decision Maker'

    def is_displayed(self):  # show this page only to the sender player if no player pass the 5 timeout condition
        return self.player.id_in_group == 1  # and not self.session.vars['remove_player']

    def vars_for_template(self):
        round_parameters = self.player.participant.vars['problem_parameters'].loc[self.round_number-1]
        prob = round_parameters['P']
        if round_parameters['Y'] == 0:
            y = 0
        else:
            y = (-1) * round_parameters['Y']
        return {
            'x': round_parameters['X'],
            'y': y,
            'round_number': self.round_number,
            'image_path': 'first_exp/chart_prob_' + str(prob) + '.png'
        }


class ReceiverWaitPage(WaitPage):

    # title_text = "This is a wait page for the sender"
    body_text = "Please wait for your partner to choose"

    # def is_displayed(self):  # show this page only to the sender player if no player pass the 5 timeout condition
    #     return not self.session.vars['remove_player']


class ReceiverPage(CustomMturkPage):
    template_name = 'first_fix_prob_exp/ReceiverPage.html'
    form_model = 'group'
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
            # self.participant.vars['num_timeout'] += 1
            self.group.receiver_timeout = True
            print('receiver timeout:', self.group.receiver_timeout)

        self.group.set_payoffs()  # run this only once - after the receiver choice

        # if self.participant.vars['num_timeout'] > Constants.num_timeouts_remove:
        #     self.session.vars['remove_player'] = True
        #     self.session.vars['player_to_remove'] = 'Decision Maker'
        #     self.session.vars['player_not_remove'] = 'Expert'

    def is_displayed(self):  # show this page only to the receiver player if no player pass the 5 timeout condition
        return self.player.id_in_group == 2  # and not self.session.vars['remove_player']

    def vars_for_template(self):
        expert = self.group.get_player_by_role('Expert')
        round_parameters = expert.participant.vars['problem_parameters'].loc[self.round_number-1]
        # for debug:
        print('round_parameters for round ', self.round_number, 'are: ', round_parameters)
        # if the sender had timeout --> the prob the receiver will see is ?
        print('sender_timeout:', self.group.sender_timeout)
        if self.group.sender_timeout:
            prob_for_dm = '?'
        else:
            prob_for_dm = self.group.sender_answer
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


class Results(CustomMturkPage):
    """This page displays the result of the round - what the receiver choose and what was the result of the lottery"""

    template_name = 'first_fix_prob_exp/Results.html'

    def get_timeout_seconds(self):
        if self.round_number in Constants.first_rounds:
            return 10
        else:
            return 5

    # def is_displayed(self):  # show this page only to the sender player if no player pass the 5 timeout condition
    #     return not self.session.vars['remove_player']

    def vars_for_template(self):
        if self.group.receiver_choice:  # if the receiver chose Status quo
            receiver_choice = 'Status quo'
            other_choice = 'Action'
            other_gain_receiver = abs(self.group.lottery_result)
            receiver_payoff = 0
            if self.group.lottery_result < 0:
                negative_other_gain_receiver = True
            else:
                negative_other_gain_receiver = False
            other_gain_sender = 1
        else:
            receiver_choice = 'Action'
            other_choice = 'Status quo'
            other_gain_receiver = 0
            receiver_payoff = self.group.lottery_result
            negative_other_gain_receiver = False
            other_gain_sender = 0

        print('lottery_result:', self.group.lottery_result)
        print('receiver_payoff:', receiver_payoff)
        sender_timeout = self.group.sender_timeout
        return {
            'round': self.round_number,
            'receiver_choice': receiver_choice,
            'other_choice': other_choice,
            'lottery_result': self.group.lottery_result,
            'other_gain_receiver': other_gain_receiver,
            'other_gain_sender': other_gain_sender,
            'receiver_payoff': receiver_payoff,
            'sender_payoff': self.group.sender_payoff,
            'receiver_negative_result': abs(self.group.lottery_result),
            'sender_timeout': sender_timeout,
            'negative_other_gain_receiver': negative_other_gain_receiver,
            'receiver_timeout': self.group.receiver_timeout
        }


class GameOver(CustomMturkPage):
    """
    This page will be displayed after the last round is over - the experiment is finish.
    It will display the results: the payoff of each player
    """
    template_name = 'first_fix_prob_exp/GameOver.html'

    def is_displayed(self):
        # show this page only after the last round
        return self.round_number == Constants.num_rounds  # and not self.session.vars['remove_player']

    def vars_for_template(self):
        # get the number of points of each player and convert to real money
        sender = self.group.get_player_by_role('Expert')
        if self.player.id_in_group == 2:  # calculate the receiver final payoff
            receiver = self.group.get_player_by_role('Decision Maker')
            # receiver total points
            receiver_total_points = sum([p.payoff for p in receiver.in_all_rounds()]) + \
                                    self.session.vars['initial_points_receiver']
            print('receiver_total_points', receiver_total_points)
            receiver_p_to_bonus = float(receiver_total_points / sender.participant.vars['max_points'])
            print('receiver_p_to_bonus:', receiver_p_to_bonus)
            receiver_bonus = Constants.bonus if random.random() <= receiver_p_to_bonus else 0
            print('receiver_bonus', receiver_bonus)
            receiver_total_payoff = receiver_bonus + self.session.config['participation_fee'] + \
                                    receiver.participant.vars['payment_for_wait']
            receiver_total_payoff = c(receiver_total_payoff)
            print('receiver payoff', receiver_total_payoff)

            print('max points for receiver:', sender.participant.vars['max_points'])

            self.player.participant.payoff = c(receiver_bonus + receiver.participant.vars['payment_for_wait'])
            print('receiver final payoff:', self.player.participant.payoff)

            return {'player_total_payoff': receiver_total_payoff,
                    'player_bonus': self.player.participant.payoff,
                    'participation_fee': self.session.config['participation_fee'],
                    }

        else:   # calculate the sender final payoff
            # sender total points
            sender_total_points = sum([p.payoff for p in sender.in_all_rounds()])
            print('sender_total_points', sender_total_points)
            sender_p_to_bonus = float(sender_total_points / Constants.num_rounds)
            sender_bonus = Constants.bonus if random.random() <= sender_p_to_bonus else 0
            print('sender_bonus', sender_bonus)

            sender_total_payoff = sender_bonus + self.session.config['participation_fee'] + \
                                  sender.participant.vars['payment_for_wait']
            sender_total_payoff = c(sender_total_payoff)
            print('sender_total_payoff', sender_total_payoff)
            self.player.participant.payoff = c(sender_bonus + sender.participant.vars['payment_for_wait'])
            print('sender final payoff:', self.player.participant.payoff)

            return {'player_total_payoff': sender_total_payoff,
                    'player_bonus': self.player.participant.payoff,
                    'participation_fee': self.session.config['participation_fee'],
                    }


# class PlayerRemoved(CustomMturkPage):
#     """
#     This page will be shown if one of the players pass the 5 timeouts
#     """
#     template_name = 'first_exp/PlayerRemoved.html'
#
#     def is_displayed(self):
#         # show this page if a player pass the 5 timeouts and self.round_number == Constants.num_rounds + 1
#         return self.session.vars['remove_player']
#
#     def vars_for_template(self):
#         player_not_removed = self.group.get_player_by_role(self.session.vars['player_not_remove'])
#
#         # get the number of points of each player and convert to real money
#         player_total_payoff = player_not_removed.participant.payoff_plus_participation_fee()
#         print('player_total_payoff without : payment_for_wait', player_total_payoff)
#         player_total_payoff += self.participant.vars['payment_for_wait']
#         return {
#             'player_removed': self.session.vars['player_to_remove'],
#             'player_total_payoff': player_total_payoff,
#             'removed_player_total_payoff': c(0).to_real_world_currency(self.session)
#         }


class OnePlayerWait(Page):
    """
    This page will be shown if only one player get into the experiment, and after 10 minutes decided to quit
    """
    template_name = 'first_fix_prob_exp/OnePlayerWait.html'

    def is_displayed(self):
        # This page inherits only from Page not from CustomMturkPage: Will appear even to players who have hit the
        # "finish study button" on a CustomMturkWaitPage skip_until_the_end_of
        return self.round_number == Constants.num_rounds and self.player.participant.vars.get('go_to_the_end')

    def vars_for_template(self):
        print('finish the study')
        self.player.participant.payoff = self.player.participant.vars['payment_for_wait'] -\
                                         self.session.config['participation_fee']

        return {
            'payment': round(self.player.participant.vars.get('payment_for_wait'), 2)
        }


page_sequence = [
    GroupedWaitPage,
    Introduction,
    PersonalInformation,
    AfterInstructions,
    SenderPage,
    # StartWP,
    ReceiverWaitPage,
    ReceiverPage,
    # SenderWaitPage,
    ReceiverWaitPage,
    Results,
    GameOver,
    OnePlayerWait,
]
