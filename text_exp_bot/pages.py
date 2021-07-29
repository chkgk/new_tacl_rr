from otree.api import Currency as c, currency_range
import random
from otree_mturk_utils.pages import CustomMturkPage, CustomMturkWaitPage
from ._builtin import Page, WaitPage
from .models import Constants
from mcts_.mcts_main import mcts_live_simu
import time
import gc
import pandas as pd
import threading

# action = {}
# in_thred = {0:0}
# in_mcts = False
# finish_mcts = {0: False, 1:False, 2:False,3:False,4:False,5:False, 6:False,7:False,8:False,9:False,10:False}
# receiver_finish_round = {0: False, 1:False, 2:False,3:False,4:False,5:False, 6:False,7:False,8:False,9:False,10:False}
# class GroupedWaitPage(CustomMturkWaitPage):
#     group_by_arrival_time = True
#     pay_by_time = Constants.pay_by_time_for_waiting / 60  # they will get 9 cents for a minute they wait
#     startwp_timer = 360  # after 6 minutes they can skip to the end of the experiment
#     startwp_timer_to_timeout = 600  # after 10 minutes the experiment will automatically end
#     use_task = False
#
#     def is_displayed(self):
#         return self.round_number == 1
#


class Introduction(CustomMturkPage):
    template_name = 'text_exp_bot/Introduction.html'
    form_model = 'player'
    form_fields = ['intro_test']

    def get_timeout_seconds(self):
        return Constants.intro_timeout_minutes*60

    def is_displayed(self):
        return self.round_number == 1  # show this page only in round 1: in the beginning of the game

    def before_next_page(self):
        if self.timeout_happened:  # if the participant didn't read the instructions after 10 minutes
            self.participant.vars['instruction_timeout'] = True
            self.group.instruction_timeout = True
            for group_round in self.group.in_rounds(1, Constants.num_rounds):
                group_round.instruction_timeout = True

        else:  # no timeout
            if str.lower(self.player.intro_test) != 'sdkot':
                # print('participant did not answer correctly on the intro test')
                self.participant.vars['failed_intro_test'] = True
                self.group.failed_intro_test = True
                for group_round in self.group.in_rounds(1, Constants.num_rounds):
                    group_round.failed_intro_test = True

            else:
                self.participant.vars['failed_intro_test'] = False

    def vars_for_template(self):
        if self.player.id_in_group == 1:  # run this once
            self.subsession.creating_session()
        if self.player.id_in_group == 1:
            other_role = 'Decision Maker'
        else:
            other_role = 'Expert'

        num_condition =\
            True if self.session.config['cond'] == 'num' or self.session.config['cond'] == 'only_num' else False
        only_num_condition = True if self.session.config['cond'] == 'only_num' else False
        only_text_condition = True if self.session.config['cond'] == 'only_text' else False
        both_condition = True if self.session.config['cond'] == 'both' else False

        return {
            'participation_fee': Constants.real_participation_fee,
            'other_role': other_role,
            'initial_points_dc': self.session.vars['initial_points_receiver'],
            'num_condition': num_condition,
            'both_condition': both_condition,
            'only_num_condition': only_num_condition,
            'only_text_condition': only_text_condition,
        }


class AfterIntroTest(WaitPage):
    def is_displayed(self):  # show this page only in round 1: in the beginning of the game
        return self.round_number == 1 and not self.group.failed_intro_test


class IntroTimeout(CustomMturkPage):
    template_name = 'text_exp_bot/IntroTimeout.html'

    def get_timeout_seconds(self):
        return 30

    def before_next_page(self):
        self.player.participant.vars['already_saw_IntroTimeout_page'] = True
        self.player.participant.vars['automate_timeout'] = True

    def is_displayed(self):
        # show this page only in round 1 and only if one or both of the players had timeout in the instructions
        if not self.player.participant.vars.get('go_to_the_end', False):  # players who didn't get a partner
            return self.round_number == 1 and self.group.instruction_timeout and not\
                self.player.participant.vars.get('already_saw_IntroTimeout_page', False)

        else:
            return False

    def vars_for_template(self):
        # tell the participant that had timeout that and the participant that passed but the partner had timeout
        # that his partner failed and we will pay him for waiting
        sender = self.group.get_player_by_role('Expert')
        receiver = self.group.get_player_by_role('Decision Maker')
        sender_timeout = sender.participant.vars.get('instruction_timeout')
        receiver_timeout = receiver.participant.vars.get('instruction_timeout')
        if self.player.id_in_group == 1:
            if sender_timeout:  # the participant had timeout
                participant_timeout = True
                sender.participant.payoff = c(0)
            else:
                participant_timeout = False
                sender.participant.payoff = c(0 + sender.participant.vars['payment_for_wait'])
            if receiver_timeout:  # the participant didn't had timeout but his partner di
                partner_timeout = True
            else:
                partner_timeout = False
        else:  # decision maker
            if receiver_timeout:  # the participant filed the test
                participant_timeout = True
                receiver.participant.payoff = c(0)
            else:
                participant_timeout = False
                receiver.participant.payoff = c(0 + receiver.participant.vars['payment_for_wait'])
            if sender_timeout:  # the participant didn't filed but his partner failed
                partner_timeout = True
            else:
                partner_timeout = False

        if self.group.pass_intro_test:
            page_name = 'insert your personal information'
        else:
            page_name = 'read the instructions'

        return {
            'participant_timeout': participant_timeout,
            'partner_timeout': partner_timeout,
            'page_name': page_name,
        }


class IntroTestFeedback(CustomMturkPage):
    template_name = 'text_exp_bot/IntroTestFeedback.html'

    def get_timeout_seconds(self):
        return 30

    def is_displayed(self):
        # show this page only in round 1: in the beginning of the game and only if one or both of the players failed
        #if not self.player.participant.vars.get('go_to_the_end', False):  # players who didn't get a partner
        #    return self.round_number == 1 and self.group.failed_intro_test and not self.group.instruction_timeout\
        #           and not self.player.participant.vars.get('already_saw_IntroTestFeedback_page', False)
        if self.group.failed_intro_test:
            return True
        else:
            return False

    def before_next_page(self):
        self.player.participant.vars['already_saw_IntroTestFeedback_page'] = True
        self.player.participant.vars['automate_timeout'] = True

    def vars_for_template(self):
        # tell the participant that failed that he failed and the participant that passed but the partner failed
        # that his partner failed and we will pay him for waiting
        #sender = self.group.get_player_by_role('Expert')
        receiver = self.group.get_player_by_role('Decision Maker')
        #sender_failed = sender.participant.vars.get('failed_intro_test')
        receiver_failed = receiver.participant.vars.get('failed_intro_test')
        #if self.player.id_in_group == 1:
            #if sender_failed:  # the participant filed the test
        participant_filed = True
        receiver.participant.payoff = c(0)
            #else:
        #participant_filed = False
        #        sender.participant.payoff = c(0 + sender.participant.vars['payment_for_wait'])
        #if receiver_failed:  # the participant didn't filed but his partner failed
        partner_filed = False

        #else:  # decision maker
        #    if receiver_failed:  # the participant filed the test
        #        participant_filed = True
        receiver.participant.payoff = c(0)
        #    else:
        #        participant_filed = False
        #        receiver.participant.payoff = c(0 + receiver.participant.vars['payment_for_wait'])
         #   if sender_failed:  # the participant didn't filed but his partner failed
         #       partner_filed = True
         #   else:
         #       partner_filed = False

        return {
            'participant_filed': participant_filed,
            'partner_filed': partner_filed,
        }


class PersonalInformation(CustomMturkPage):
    template_name = 'text_exp_bot/PersonalInformation.html'
    form_model = 'player'
    form_fields = ['name', 'age', 'gender', 'is_student', 'occupation', 'residence']

    def get_timeout_seconds(self):
        return 240

    def is_displayed(self):  # show this page only in round 1: in the beginning of the game
        if not self.player.participant.vars.get('go_to_the_end', False):  # players who didn't get a partner
            return self.round_number == 1 and not self.group.instruction_timeout and not self.group.failed_intro_test

        else:
            return False

    def before_next_page(self):
        self.group.pass_intro_test = True  # if we are here --> they pass the intro test
        if self.timeout_happened:  # if the participant didn't read the instructions after 10 minutes
            self.participant.vars['instruction_timeout'] = True
            self.group.instruction_timeout = True
            for group_round in self.group.in_rounds(1, Constants.num_rounds):
                group_round.instruction_timeout = True
        else:
            self.participant.vars['name'] = self.player.name
            self.participant.vars['age'] = self.player.age
            self.participant.vars['gender'] = self.player.gender
            self.participant.vars['occupation'] = self.player.occupation
            self.participant.vars['is_student'] = self.player.is_student
            self.participant.vars['residence'] = self.player.residence
            self.participant.vars['num_timeout'] = 0


class AfterInstructions(WaitPage):
    def is_displayed(self):  # show this page only in round 1: in the beginning of the game
        # players_failed_intro = list()
        # for p in self.group.get_players():
        #     if p.participant.vars.get('go_to_the_end', False):  # players who didn't get a partner
        #         return False
        #     else:
        #         players_failed_intro.append(p.participant.vars.get('failed_intro_test'))
        return self.round_number == 1 and not self.group.instruction_timeout and not self.group.failed_intro_test


class SenderPage(CustomMturkPage):
    template_name = 'text_exp_bot/SenderPage.html'
    form_model = 'group'
    form_fields = ['sender_answer_index']

    timeout_submission = {'sender_timeout': True}
    # if the sender didn't provide a probability after seconds_wait,
    # the probability we will show to the receiver will be ?

    def get_timeout_seconds(self):
        if self.round_number in Constants.first_rounds:
            return Constants.seconds_wait_first_rounds_expert
        else:
            return Constants.seconds_wait_expert

    def before_next_page(self):
        if self.timeout_happened:  # if the sender didn't provide a probability after seconds_wait
            self.group.sender_timeout = True
            # if the expert had timeout- the worst review will be chosen
        self.group.sender_answer_index = 7

        round_parameters = self.player.participant.vars['problem_parameters'].loc[self.round_number-1]
        self.group.sender_answer_scores = round_parameters[f'score_{self.group.sender_answer_index-1}']
        self.group.sender_answer_reviews =\
            round_parameters[f'random_positive_negative_review_{self.group.sender_answer_index-1}']
        self.group.sender_answer_positive_reviews =\
            round_parameters[f'positive_review_{self.group.sender_answer_index-1}']
        self.group.sender_answer_negative_reviews =\
            round_parameters[f'negative_review_{self.group.sender_answer_index-1}']

    def is_displayed(self):  # show this page only to the sender player if no player pass the 5 timeout condition
        if not self.player.participant.vars.get('go_to_the_end', False):  # players who didn't get a partner
            #print(f'SenderPage: self.group.instruction_timeout: {self.group.instruction_timeout}')
            #print(f'SenderPage: self.group.failed_intro_test: {self.group.failed_intro_test}')

            return self.player.id_in_group == 1 and not self.group.instruction_timeout and not\
                self.group.failed_intro_test
        else:
            return False

    def vars_for_template(self):
        self.group.set_round_parameters()
        round_parameters = self.player.participant.vars['problem_parameters'].loc[self.round_number-1]
        scores = list()
        negative_positive_reviews = list()
        reviews = list()
        condition = self.session.config['cond']
        for i in range(Constants.number_reviews_per_hotel):
            negative_positive_reviews.append(round_parameters[f'random_positive_negative_review_{i}'])
            reviews.append(round_parameters[f'review_{i}'])
            scores.append(round_parameters[f'score_{i}'])

    #     # print(f'the negative_positive_reviews is: {negative_positive_reviews}\n and reviews is: {reviews}')

        return {
            'round_number': self.round_number,
            'condition': condition,
            'scores': scores,
            'negative_positive_reviews': negative_positive_reviews,
            'reviews': reviews,
        }


def maya(df,round):
    return mcts_live_simu(df,round)


class ReceiverWaitPage(Page):
    template_name = 'text_exp_bot/ReceiverWaitPage.html'
    timeout_seconds = 200

    def before_next_page(self):
        # print(sum([1 for key,val in finish_mcts.items() if val ==True]), self.group.round_number-1)
        # receiver_finish_round[self.group.round_number-1] == finish_mcts[self.group.round_number-1] and  in_thred[0] == self.group.round_number - 1 and finish_mcts[self.group.round_number-1] == False and
        # if (self.group.round_number - 1) == sum([1 for key, val in finish_mcts.items() if val == True]):
        if not self.group.finish_mcts:
            #in_thred[0] = in_thred[0] + 1
            # finish_mcts[self.group.round_number-1] = True
            self.group.finish_mcts = True
            print(f'enter to "if"')
            if self.group.round_number == 1:
                new_df = pd.DataFrame(columns=self.player.participant.vars['df'].columns)
                for round_ in range(1, 11):
                    # print(self.participant.vars['problem_parameters']['average_score'],'lplp')
                    new_df.at[round_ - 1, 'player_id_in_group'] = 2
                    new_df.at[round_ - 1, 'status'] = 'play'
                    new_df.at[round_ - 1, 'group_negative_review_0'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'negative_review_0']
                    new_df.at[round_ - 1, 'group_negative_review_1'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'negative_review_1']
                    new_df.at[round_ - 1, 'group_negative_review_2'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'negative_review_2']
                    new_df.at[round_ - 1, 'group_negative_review_3'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'negative_review_3']
                    new_df.at[round_ - 1, 'group_negative_review_4'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'negative_review_4']
                    new_df.at[round_ - 1, 'group_negative_review_5'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'negative_review_5']
                    new_df.at[round_ - 1, 'group_negative_review_6'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'negative_review_6']
                    new_df.at[round_ - 1, 'group_positive_review_5'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'positive_review_5']
                    self.player.participant.vars['df'].at[round_ - 1, 'group_positive_review_4'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'positive_review_4']
                    new_df.at[round_ - 1, 'group_positive_review_3'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'positive_review_3']
                    new_df.at[round_ - 1, 'group_positive_review_1'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'positive_review_1']
                    new_df.at[round_ - 1, 'group_positive_review_2'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'positive_review_2']
                    new_df.at[round_ - 1, 'group_positive_review_0'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'positive_review_0']
                    new_df.at[round_ - 1, 'group_random_positive_negative_review_0'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'random_positive_negative_review_0']
                    new_df.at[round_ - 1, 'group_random_positive_negative_review_1'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'random_positive_negative_review_1']
                    new_df.at[round_ - 1, 'group_random_positive_negative_review_2'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'random_positive_negative_review_2']
                    new_df.at[round_ - 1, 'group_random_positive_negative_review_3'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'random_positive_negative_review_3']
                    new_df.at[round_ - 1, 'group_random_positive_negative_review_4'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'random_positive_negative_review_4']
                    new_df.at[round_ - 1, 'group_random_positive_negative_review_5'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'random_positive_negative_review_5']
                    new_df.at[round_ - 1, 'group_random_positive_negative_review_6'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'random_positive_negative_review_6']
                    new_df.at[round_ - 1, 'group_score_0'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'score_0']
                    new_df.at[round_ - 1, 'group_score_1'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'score_1']
                    new_df.at[round_ - 1, 'group_score_2'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'score_2']
                    new_df.at[round_ - 1, 'group_score_3'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'score_3']
                    new_df.at[round_ - 1, 'group_score_4'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'score_4']
                    new_df.at[round_ - 1, 'group_score_5'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'score_5']
                    new_df.at[round_ - 1, 'group_score_6'] = \
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'score_6']
                    new_df.at[
                        round_ - 1, 'subsession_round_number'] = round_  # average_score
                    new_df.at[round_ - 1, 'group_average_score'] = round(
                        self.participant.vars['problem_parameters'].at[round_ - 1, 'average_score'],
                        2)  # round(self.group.average_score,2)
                # action[self.group.round_number] = mcts_live_simu(new_df, self.group.round_number)
                self.group.action = mcts_live_simu(new_df, self.group.round_number)
            else:
                # action[self.group.round_number] = mcts_live_simu(self.player.participant.vars['df'], self.group.round_number)
                self.group.action = mcts_live_simu(self.player.participant.vars['df'], self.group.round_number)
            # finish_mcts[self.group.round_number] = True
        else:
            print('enter to "else"')
            if self.group.round_number == 1:
                for round_ in range(1, 11):
                    #print(self.participant.vars['problem_parameters']['average_score'],'lplp')
                    self.player.participant.vars['df'].at[round_ - 1,'player_id_in_group'] = 2
                    self.player.participant.vars['df'].at[round_ - 1,'status'] = 'play'
                    self.player.participant.vars['df'].at[round_ - 1,'group_negative_review_0'] = self.participant.vars['problem_parameters'].at[round_ - 1,'negative_review_0']
                    self.player.participant.vars['df'].at[round_ - 1,'group_negative_review_1'] = self.participant.vars['problem_parameters'].at[round_ - 1,'negative_review_1']
                    self.player.participant.vars['df'].at[round_ - 1,'group_negative_review_2'] = self.participant.vars['problem_parameters'].at[round_ - 1,'negative_review_2']
                    self.player.participant.vars['df'].at[round_ - 1,'group_negative_review_3'] = self.participant.vars['problem_parameters'].at[round_ - 1,'negative_review_3']
                    self.player.participant.vars['df'].at[round_ - 1,'group_negative_review_4'] = self.participant.vars['problem_parameters'].at[round_ - 1,'negative_review_4']
                    self.player.participant.vars['df'].at[round_ - 1,'group_negative_review_5'] = self.participant.vars['problem_parameters'].at[round_ - 1,'negative_review_5']
                    self.player.participant.vars['df'].at[round_ - 1,'group_negative_review_6'] = self.participant.vars['problem_parameters'].at[round_ - 1,'negative_review_6']
                    self.player.participant.vars['df'].at[round_ - 1,'group_positive_review_5'] = self.participant.vars['problem_parameters'].at[round_ - 1,'positive_review_5']
                    self.player.participant.vars['df'].at[round_ - 1,'group_positive_review_4'] = self.participant.vars['problem_parameters'].at[round_ - 1,'positive_review_4']
                    self.player.participant.vars['df'].at[round_ - 1,'group_positive_review_3'] = self.participant.vars['problem_parameters'].at[round_ - 1,'positive_review_3']
                    self.player.participant.vars['df'].at[round_ - 1,'group_positive_review_1'] = self.participant.vars['problem_parameters'].at[round_ - 1,'positive_review_1']
                    self.player.participant.vars['df'].at[round_ - 1,'group_positive_review_2'] = self.participant.vars['problem_parameters'].at[round_ - 1,'positive_review_2']
                    self.player.participant.vars['df'].at[round_ - 1,'group_positive_review_0'] = self.participant.vars['problem_parameters'].at[round_ - 1,'positive_review_0']
                    self.player.participant.vars['df'].at[round_ - 1,'group_random_positive_negative_review_0'] = self.participant.vars['problem_parameters'].at[round_ - 1,'random_positive_negative_review_0']
                    self.player.participant.vars['df'].at[round_ - 1,'group_random_positive_negative_review_1'] = self.participant.vars['problem_parameters'].at[round_ - 1,'random_positive_negative_review_1']
                    self.player.participant.vars['df'].at[round_ - 1,'group_random_positive_negative_review_2'] = self.participant.vars['problem_parameters'].at[round_ - 1,'random_positive_negative_review_2']
                    self.player.participant.vars['df'].at[round_ - 1,'group_random_positive_negative_review_3'] = self.participant.vars['problem_parameters'].at[round_ - 1,'random_positive_negative_review_3']
                    self.player.participant.vars['df'].at[round_ - 1,'group_random_positive_negative_review_4'] = self.participant.vars['problem_parameters'].at[round_ - 1,'random_positive_negative_review_4']
                    self.player.participant.vars['df'].at[round_ - 1,'group_random_positive_negative_review_5'] = self.participant.vars['problem_parameters'].at[round_ - 1,'random_positive_negative_review_5']
                    self.player.participant.vars['df'].at[round_ - 1,'group_random_positive_negative_review_6'] = self.participant.vars['problem_parameters'].at[round_ - 1,'random_positive_negative_review_6']
                    self.player.participant.vars['df'].at[round_ - 1,'group_score_0'] = self.participant.vars['problem_parameters'].at[round_ - 1,'score_0']
                    self.player.participant.vars['df'].at[round_ - 1,'group_score_1'] = self.participant.vars['problem_parameters'].at[round_ - 1,'score_1']
                    self.player.participant.vars['df'].at[round_ - 1,'group_score_2'] = self.participant.vars['problem_parameters'].at[round_ - 1,'score_2']
                    self.player.participant.vars['df'].at[round_ - 1,'group_score_3'] = self.participant.vars['problem_parameters'].at[round_ - 1,'score_3']
                    self.player.participant.vars['df'].at[round_ - 1,'group_score_4'] = self.participant.vars['problem_parameters'].at[round_ - 1,'score_4']
                    self.player.participant.vars['df'].at[round_ - 1,'group_score_5'] = self.participant.vars['problem_parameters'].at[round_ - 1,'score_5']
                    self.player.participant.vars['df'].at[round_ - 1,'group_score_6'] = self.participant.vars['problem_parameters'].at[round_ - 1,'score_6']
                    self.player.participant.vars['df'].at[round_ - 1, 'subsession_round_number'] = round_#average_score
                    self.player.participant.vars['df'].at[round_ - 1, 'group_average_score'] = round(self.participant.vars['problem_parameters'].at[round_ - 1,'average_score'],2)#round(self.group.average_score,2)
            try:
                print('after_mcts!')
                # self.group.sender_answer_index = action[self.group.round_number] if action[self.group.round_number]!=None else 6
                self.group.sender_answer_index = self.group.action if self.group.action is not None else 6
                print(f'mcts_result:{self.group.sender_answer_index}')
                round_parameters = self.player.participant.vars['problem_parameters'].loc[self.group.round_number - 1]
                self.group.sender_answer_scores = round_parameters[f'score_{self.group.sender_answer_index}']
                self.group.sender_answer_reviews = \
                    round_parameters[f'random_positive_negative_review_{self.group.sender_answer_index}']
                print(f'mcts_rev: {self.group.sender_answer_reviews}')
                self.group.sender_answer_positive_reviews = \
                    round_parameters[f'positive_review_{self.group.sender_answer_index}']
                self.group.sender_answer_negative_reviews = \
                    round_parameters[f'negative_review_{self.group.sender_answer_index}']
                print('table results are updated!')
            except:
                None

    def is_displayed(self):
        # if self.group.round_number not in action.keys():
        if self.group.action == 10:  # this round has not run yet
            self.group.set_round_parameters()
            self.before_next_page()
            #print('ReceiverWaitPage is_displayed')
#             x = threading.Thread(target=self.before_next_page)
#             x.start()
            #print('continue to get timeout')
        if self.group.failed_intro_test != True:
            return True

    def get_timeout_seconds(self):
        print('timeout begin')
        return 200


    # def before_next_page(self):
    #     print('ReceiverWaitPage before_next_page')
    #     start_time = time.time()
    #     print('begin self.group.set_round_parameters()')
    #     self.group.set_round_parameters()
    #     print(f"--- self.group.set_round_parameters ---{time.time() - start_time}")
    #     start_time = time.time()
    #
    #     #print('mayaaaa')
    #     #print(self.group.random_positive_negative_review_0)
    #     if self.group.round_number==1:
    #         for round_ in range(1,11):
    #             #print(self.participant.vars['problem_parameters']['average_score'],'lplp')
    #             self.player.participant.vars['df'].at[round_ - 1,'player_id_in_group'] = 2
    #             self.player.participant.vars['df'].at[round_ - 1,'status'] = 'play'
    #             self.player.participant.vars['df'].at[round_ - 1,'group_negative_review_0'] = self.participant.vars['problem_parameters'].at[round_ - 1,'negative_review_0']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_negative_review_1'] = self.participant.vars['problem_parameters'].at[round_ - 1,'negative_review_1']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_negative_review_2'] = self.participant.vars['problem_parameters'].at[round_ - 1,'negative_review_2']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_negative_review_3'] = self.participant.vars['problem_parameters'].at[round_ - 1,'negative_review_3']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_negative_review_4'] = self.participant.vars['problem_parameters'].at[round_ - 1,'negative_review_4']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_negative_review_5'] = self.participant.vars['problem_parameters'].at[round_ - 1,'negative_review_5']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_negative_review_6'] = self.participant.vars['problem_parameters'].at[round_ - 1,'negative_review_6']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_positive_review_5'] = self.participant.vars['problem_parameters'].at[round_ - 1,'positive_review_5']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_positive_review_4'] = self.participant.vars['problem_parameters'].at[round_ - 1,'positive_review_4']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_positive_review_3'] = self.participant.vars['problem_parameters'].at[round_ - 1,'positive_review_3']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_positive_review_1'] = self.participant.vars['problem_parameters'].at[round_ - 1,'positive_review_1']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_positive_review_2'] = self.participant.vars['problem_parameters'].at[round_ - 1,'positive_review_2']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_positive_review_0'] = self.participant.vars['problem_parameters'].at[round_ - 1,'positive_review_0']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_random_positive_negative_review_0'] = self.participant.vars['problem_parameters'].at[round_ - 1,'random_positive_negative_review_0']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_random_positive_negative_review_1'] = self.participant.vars['problem_parameters'].at[round_ - 1,'random_positive_negative_review_1']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_random_positive_negative_review_2'] = self.participant.vars['problem_parameters'].at[round_ - 1,'random_positive_negative_review_2']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_random_positive_negative_review_3'] = self.participant.vars['problem_parameters'].at[round_ - 1,'random_positive_negative_review_3']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_random_positive_negative_review_4'] = self.participant.vars['problem_parameters'].at[round_ - 1,'random_positive_negative_review_4']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_random_positive_negative_review_5'] = self.participant.vars['problem_parameters'].at[round_ - 1,'random_positive_negative_review_5']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_random_positive_negative_review_6'] = self.participant.vars['problem_parameters'].at[round_ - 1,'random_positive_negative_review_6']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_score_0'] = self.participant.vars['problem_parameters'].at[round_ - 1,'score_0']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_score_1'] = self.participant.vars['problem_parameters'].at[round_ - 1,'score_1']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_score_2'] = self.participant.vars['problem_parameters'].at[round_ - 1,'score_2']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_score_3'] = self.participant.vars['problem_parameters'].at[round_ - 1,'score_3']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_score_4'] = self.participant.vars['problem_parameters'].at[round_ - 1,'score_4']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_score_5'] = self.participant.vars['problem_parameters'].at[round_ - 1,'score_5']
    #             self.player.participant.vars['df'].at[round_ - 1,'group_score_6'] = self.participant.vars['problem_parameters'].at[round_ - 1,'score_6']
    #             self.player.participant.vars['df'].at[round_ - 1, 'subsession_round_number'] = round_#average_score
    #             self.player.participant.vars['df'].at[round_ - 1, 'group_average_score'] = round(self.participant.vars['problem_parameters'].at[round_ - 1,'average_score'],2)#round(self.group.average_score,2)
    #     self.group.sender_answer_index = maya(self.player.participant.vars['df'],self.group.round_number)
    #     print(f'mcts_result:{self.group.sender_answer_index}')
    #     round_parameters = self.player.participant.vars['problem_parameters'].loc[self.group.round_number - 1]
    #     #print(round_parameters)
    #     self.group.sender_answer_scores = round_parameters[f'score_{self.group.sender_answer_index }']
    #     self.group.sender_answer_reviews = \
    #         round_parameters[f'random_positive_negative_review_{self.group.sender_answer_index }']
    #     self.group.sende
    #     r_answer_positive_reviews = \
    #         round_parameters[f'positive_review_{self.group.sender_answer_index }']
    #     self.group.sender_answer_negative_reviews = \
    #         round_parameters[f'negative_review_{self.group.sender_answer_index }']
    #     #not self.group.failed_intro_test and not self.group.instruction_timeout
    # #print("--- %s seconds ---" % (time.time() - start_time))

class ReceiverPage(CustomMturkPage):
    template_name = 'text_exp_bot/ReceiverPage.html'
    form_model = 'group'
    form_fields = ['receiver_choice']

    timeout_submission = {'receiver_choice': bool(random.getrandbits(1))}
    # if the receiver didn't choose his choice after seconds_wait, an option will be chosen randomly

    def is_displayed(self):
        if self.group.failed_intro_test != True:
            return True

    def get_timeout_seconds(self):
        # print('ReceiverPage get_timeout_seconds')
        if self.round_number in Constants.first_rounds:
            return Constants.seconds_wait_first_rounds_dm
        else:
            return Constants.seconds_wait_dm

    def before_next_page(self):
        # print('ReceiverPage before_next_page')
        if self.timeout_happened:  # if the receiver didn't provide a probability after seconds_wait
            # self.participant.vars['num_timeout'] += 1
            self.group.receiver_timeout = True
            # print('receiver timeout:', self.group.receiver_timeout)
        self.group.set_payoffs()

    def vars_for_template(self):
        # print('ReceiverPage vars_for_template')

        if self.session.config['cond'] == 'verbal' or self.session.config['cond'] == 'only_text':
            score_all_review = self.group.sender_answer_reviews
            expert_selected = 'review'
            only_score = self.group.sender_answer_scores
        elif self.session.config['cond'] == 'both':
            score_all_review = self.group.sender_answer_reviews
            only_score = self.group.sender_answer_scores
            expert_selected = 'review and score'
        else:  # num + only_num
            score_all_review = self.group.sender_answer_scores
            expert_selected = 'score'
            only_score = self.group.sender_answer_scores
        # print('prob:', score_review)
        return {
            'score_all_review': score_all_review,
            'only_score': only_score,
            'round_number': self.round_number,
            'expert_selected': expert_selected,
            'sender_timeout': self.group.receiver_timeout,
        }


class Results(CustomMturkPage):
    """This page displays the result of the round - what the receiver choose and what was the result of the lottery"""

    template_name = 'text_exp_bot/Results.html'

    def is_displayed(self):
        if self.group.failed_intro_test != True:
            return True

    def get_timeout_seconds(self):
        if self.round_number in Constants.first_rounds or self.round_number == Constants.num_rounds:
            return 30
        else:
            return 10

    def before_next_page(self):
        # receiver_finish_round[self.group.round_number - 1] = True
        self.group.receiver_finish_round = True
    # def is_displayed(self):  # show this page only to the sender player if no player pass the 5 timeout condition
    #     if not self.player.participant.vars.get('go_to_the_end', False):  # players who didn't get a partner
    #         return not self.group.instruction_timeout and not self.group.failed_intro_test
    #     else:
    #         return False

    def vars_for_template(self):
        self.player.participant.vars['df'].at[self.group.round_number-1,'subsession_round_number']=self.group.round_number
        self.player.participant.vars['df'].at[self.group.round_number-1,'group_receiver_payoff']=self.group.receiver_payoff
        self.player.participant.vars['df'].at[self.group.round_number-1,'group_sender_payoff']=(1 if self.group.receiver_payoff!=0 else 0)
        self.player.participant.vars['df'].at[self.group.round_number-1,'chose_lose']=(1 if self.group.receiver_payoff<0 else 0)
        self.player.participant.vars['df'].at[self.group.round_number-1,'chose_earn']=(1 if self.group.receiver_payoff>0 else 0)
        self.player.participant.vars['df'].at[self.group.round_number-1,'not_chose_lose']=(1 if self.group.receiver_payoff==0 and None else 0)
        self.player.participant.vars['df'].at[self.group.round_number-1,'not_chose_earn']=(1 if self.group.receiver_payoff==0 and None else 0)
        self.player.participant.vars['df'].at[self.group.round_number-1,'group_sender_answer_scores']=self.group.sender_answer_scores
        self.player.participant.vars['df'].at[self.group.round_number-1,'group_sender_answer_index']=self.group.sender_answer_index
        self.player.participant.vars['df'].at[self.group.round_number-1,'group_average_score']=round(self.group.average_score,2)
        self.player.participant.vars['df'].at[self.group.round_number-1,'group_sender_answer_reviews']=self.group.sender_answer_reviews
        self.player.participant.vars['df'].at[self.group.round_number-1,'group_lottery_result']=self.group.lottery_result
        self.player.participant.vars['df'].at[self.group.round_number-1,'lottery_result_low']=(1 if self.group.lottery_result<3 else 0)
        self.player.participant.vars['df'].at[self.group.round_number-1,'lottery_result_med1'] = (1 if (self.group.lottery_result>=3 and self.group.lottery_result<=5) else 0)
        self.player.participant.vars['df'].at[self.group.round_number-1,'lottery_result_high']=(1 if self.group.lottery_result>=8 else 0)
        reviews_features = pd.read_csv('mcts_/hotels_index_test_data.csv')
        if self.player.participant.vars['df'].at[self.group.round_number-1,'group_sender_answer_reviews'][0] == 'P':
            self.player.participant.vars['df'].at[self.group.round_number-1,'review_id']= reviews_features[(reviews_features.hotel == self.player.participant.vars['df'].at[self.group.round_number-1,'group_average_score']) & (reviews_features.rev_index == self.player.participant.vars['df'].at[self.group.round_number-1,'group_sender_answer_index']) & (
                    reviews_features.posorneg == 'Positive')]['review_id'].values[0]

        else:
            self.player.participant.vars['df'].at[self.group.round_number-1,'review_id']=reviews_features[(reviews_features.hotel == self.player.participant.vars['df'].at[self.group.round_number-1,'group_average_score']) & (reviews_features.rev_index == self.player.participant.vars['df'].at[self.group.round_number-1,'group_sender_answer_index']) & (
                reviews_features.posorneg == 'Negative')]['review_id'].values[0]
        #print(self.player.participant.vars['df'].at[self.group.round_number-1,'review_id'],'maya_test')

        if self.group.receiver_choice:  # if the receiver chose Status quo
            receiver_choice = 'Stay at home'
            other_choice = 'Hotel'
            #print('self.group.lottery_result: ', self.group.lottery_result)
            other_gain_receiver = self.group.lottery_result - Constants.cost
            receiver_payoff = 0
            if other_gain_receiver < 0:
                negative_other_gain_receiver = True
            else:
                negative_other_gain_receiver = False
            other_gain_sender = 1
        else:
            receiver_choice = 'Hotel'
            other_choice = 'Stay at home'
            other_gain_receiver = 0
            receiver_payoff = self.group.receiver_payoff
            negative_other_gain_receiver = False
            other_gain_sender = 0

        # print('lottery_result:', self.group.lottery_result)
        # print('receiver_payoff:', receiver_payoff)
        sender_timeout = self.group.sender_timeout
        return {
            'round': self.round_number,
            'receiver_choice': receiver_choice,
            'other_choice': other_choice,
            'lottery_result': self.group.lottery_result,
            'other_gain_receiver': round(abs(other_gain_receiver), 2),
            'other_gain_sender': other_gain_sender,
            'receiver_payoff': round(receiver_payoff, 2),
            'sender_payoff': self.group.sender_payoff,
            'receiver_negative_result': abs(round(self.group.receiver_payoff, 2)),
            'sender_timeout': sender_timeout,
            'negative_other_gain_receiver': round(negative_other_gain_receiver, 2),
            'receiver_timeout': self.group.receiver_timeout
        }


class Test(CustomMturkPage):
    """
    This page will be displayed only to the DM in the verbal condition, in order to test them if they read the texts.
    """
    template_name = 'text_exp_bot/Test.html'
    form_model = 'player'
    form_fields = ['dm_test_chosen_review_1', 'dm_test_chosen_review_2',
                   'dm_test_not_chosen_review_1', 'dm_test_not_chosen_review_2']

    def is_displayed(self):
        if not self.player.participant.vars.get('go_to_the_end', False):  # players who didn't get a partner
            # show this page only after the last round and only to DM in the verbal condition
            return self.round_number == Constants.num_rounds  and\
                   not self.group.instruction_timeout and not self.group.failed_intro_test
        # and self.session.config['cond'] == 'verbal'
        else:
            return False

    def before_next_page(self):
        receiver = self.group.get_player_by_role('Decision Maker')
        # the DM need to answered correctly to all texts
        chosen_reviews_sum = int(receiver.dm_test_chosen_review_1) + int(receiver.dm_test_chosen_review_2)
        not_chosen_reviews_sum = int(receiver.dm_test_not_chosen_review_1) + int(receiver.dm_test_not_chosen_review_2)
        if chosen_reviews_sum == 2 and not_chosen_reviews_sum == 0:  # 4 correct answers
            for p in self.group.get_players():
                p.participant.vars['pay'] = 1
            # print("receiver passed the test")
            for group_round in self.group.in_all_rounds():
                group_round.receiver_passed_test = 1
            self.group.receiver_passed_test = 1
        elif (chosen_reviews_sum == 2 and not_chosen_reviews_sum == 1) or\
                (chosen_reviews_sum == 1 and not_chosen_reviews_sum == 0):  # 3 correct answers
            for p in self.group.get_players():
                p.participant.vars['pay'] = 0.5
            for group_round in self.group.in_all_rounds():
                group_round.receiver_passed_test = 0.5
            self.group.receiver_passed_test = 0.5
        else:
            for p in self.group.get_players():
                p.participant.vars['pay'] = 0
            for group_round in self.group.in_all_rounds():
                group_round.receiver_passed_test = 0
            self.group.receiver_passed_test = 0
            receiver.participant.payoff = c(0)
            # print('receiver failed the test')
            self.player.participant.vars['failed_dm_test'] = True
            self.player.participant.vars['automate_timeout'] = True

    def vars_for_template(self):
        # get previous rounds chosen reviews and sample from not seen reviews
        # sender = self.group.get_player_by_role('Expert')
        chosen_reviews = list()
        if self.session.config['cond'] in ['verbal', 'only_text', 'both']:
            not_chosen_reviews =\
                Constants.reviews_not_seen.sample(n=len(Constants.rounds_for_reviews_in_test)).not_seen_reviews.tolist()
        else:  # numeric condition
            not_chosen_reviews = [2.4, 9.35]
        for round_number in Constants.rounds_for_reviews_in_test:
            p = self.player.in_round(round_number)
            if self.session.config['cond'] in ['verbal', 'only_text', 'both']:
                chosen_reviews.append(p.group.sender_answer_reviews)
            else:
                chosen_reviews.append(p.group.sender_answer_scores)
            # round_parameters = sender.participant.vars['problem_parameters'].loc[round_number - 1]
            # chosen_index = p.group.sender_answer_index
            # if chosen_index != 3:
            #     not_chosen_reviews.append(round_parameters[f'review_{2}'])
            # else:
            #     not_chosen_reviews.append(round_parameters[f'review_{3}'])

        # print(f'chosen reviews: {chosen_reviews} \nnot chosen reviews: {not_chosen_reviews}')

        return {
            'chosen_reviews': chosen_reviews,
            'not_chosen_reviews': not_chosen_reviews,
            'text_number': 'texts' if self.session.config['cond'] == 'verbal' else 'numbers',
        }


class FeedbackTest(CustomMturkPage):
    """
    This page will be displayed only to the DM in the verbal condition, this will be the feedback for the DM test
    """
    template_name = 'text_exp_bot/FeedbackTest.html'

    def is_displayed(self):
        if not self.player.participant.vars.get('go_to_the_end', False):  # players who didn't get a partner
            # show this page only after the last round and only to DM in the verbal condition
            return self.round_number == Constants.num_rounds and\
                   not self.group.instruction_timeout and not self.group.failed_intro_test
        # and self.session.config['cond'] == 'verbal'

        else:
            return False

    def vars_for_template(self):
        # get previous rounds chosen and not chosen reviews
        chosen_reviews = list()
        for round_number in Constants.rounds_for_reviews_in_test:
            p = self.player.in_round(round_number)
            chosen_reviews.append(p.group.sender_answer_reviews)
        receiver = self.group.get_player_by_role('Decision Maker')

        return {
            'chosen_reviews': chosen_reviews,
            'receiver_not_paid': receiver.participant.vars['pay'],
            'is_verbal': self.session.config['cond'] == 'verbal',
        }


# class GameOver(CustomMturkPage):
#     """
#     This page will be displayed after the last round is over - the experiment is finish.
#     It will display the results: the payoff of each player
#     """
#     template_name = 'text_exp_bot/GameOver.html'
#
#     def is_displayed(self):# show this page only after the last round  
#         if not self.player.participant.vars.get('go_to_the_end', False):
#             return not self.group.instruction_timeout and not self.group.failed_intro_test and self.round_number == Constants.num_rounds
#         else:
#             return False
#     # def is_displayed(self):
#     #     # show this page only after the last round
#     #     # receiver = self.group.get_player_by_role('Decision Maker')
#     #     # if this is the decision maker and he failed the test --> don't show this page.
#     #     print(f'GameOver: self.group.instruction_timeout: {self.group.instruction_timeout}')
#     #     print(f'GameOver: self.group.failed_intro_test: {self.group.failed_intro_test}')
#     #     if not self.player.participant.vars.get('go_to_the_end', False):  # players who didn't get a partner
#     #         if self.group.failed_intro_test or self.group.instruction_timeout:  # failed the intro test or had timeout
#     #             return False
#     #         elif self.round_number == Constants.num_rounds and self.player.id_in_group == 1:
#     #             return True
#     #         elif self.round_number == Constants.num_rounds and self.player.id_in_group == 2 and \
#     #                 self.group.get_player_by_role('Decision Maker').participant.vars['pay'] > 0:  # 3/4 correct answers
#     #                 # and self.session.config['cond'] == 'verbal'
#     #             return True
#             # elif self.round_number == Constants.num_rounds and self.player.id_in_group == 2 and\
#             #         not self.session.config['cond'] == 'verbal':  # numerical condition don't have final test
#             #     return True
#
#
#     def vars_for_template(self):
#         # get the number of points of each player and convert to real money
#         sender = self.group.get_player_by_role('Expert')
#         receiver = self.group.get_player_by_role('Decision Maker')
#         # if not self.session.config['cond'] == 'verbal':  # numerical condition don't have .participant.vars['pay']
#         #     receiver.participant.vars['pay'] = 1
#         if self.player.id_in_group == 2 and receiver.participant.vars['pay'] > 0:  # calculate the receiver final payoff
#             # receiver total points
#             receiver_total_points = sum([p.payoff for p in receiver.in_all_rounds()]) + \
#                                     self.session.vars['initial_points_receiver']
#             # print('receiver_total_points', receiver_total_points)
#             receiver_p_to_bonus = float(receiver_total_points / sender.participant.vars['max_points'])
#             # print('receiver_p_to_bonus:', receiver_p_to_bonus)
#             receiver_bonus = Constants.bonus if random.random() <= receiver_p_to_bonus else 0
#             #print('receiver_bonus', receiver_bonus)
#             if receiver.participant.vars['pay'] == 0.5:  # no bonus
#                 receiver_bonus = 0
#             #print("receiver_bonus_after check if receiver.participant.vars['pay'] == 0.5:", receiver_bonus)
#             receiver_total_payoff = receiver_bonus + Constants.real_participation_fee +\
#                 receiver.participant.vars['payment_for_wait']
#             receiver_total_payoff = c(receiver_total_payoff)
#             # print('receiver payoff', receiver_total_payoff)
#
#             # print('max points for receiver:', sender.participant.vars['max_points'])
#
#             self.player.participant.payoff = c(receiver_total_payoff)
#             # c(receiver_bonus + receiver.participant.vars['payment_for_wait'])
#             # print('receiver final payoff:', self.player.participant.payoff)
#
#             # if not receiver.participant.vars['pay']:
#             #     print('receiver failed the test and will not be paid')
#             #     self.player.participant.payoff = c(0)
#             #     receiver_total_payoff = c(0)
#
#             return {'player_total_payoff': receiver_total_payoff,
#                     'player_bonus': receiver_bonus,
#                     'participation_fee': Constants.real_participation_fee,
#                     'fee_for_waiting': round(receiver.participant.vars['payment_for_wait'], 2),
#                     }
#
#         else:   # calculate the sender final payoff
#             # sender total points
#             sender_total_points = sum([p.payoff for p in sender.in_all_rounds()])
#             # print('sender_total_points', sender_total_points)
#             sender_p_to_bonus = float(sender_total_points / Constants.num_rounds)
#             sender_bonus = Constants.bonus if random.random() <= sender_p_to_bonus else 0
#             # print('sender_bonus', sender_bonus)
#             # if the DM didn't pass the test and didnt entered all the rounds- don't pay bonus (we not use the data)
#             if 'pay' in self.group.get_player_by_role('Decision Maker').participant.vars:
#                 if self.group.get_player_by_role('Decision Maker').participant.vars['pay'] == 0 and\
#                         sender_total_points < Constants.num_rounds:
#                     sender_bonus = 0
#
#             sender_total_payoff = sender_bonus + Constants.real_participation_fee + \
#                                   sender.participant.vars['payment_for_wait']
#             sender_total_payoff = c(sender_total_payoff)
#             # print('sender_total_payoff', sender_total_payoff)
#             self.player.participant.payoff = c(sender_total_payoff)
#             # c(sender_bonus + sender.participant.vars['payment_for_wait'])
#             # print('sender final payoff:', self.player.participant.payoff)
#
#             return {'player_total_payoff': sender_total_payoff,
#                     'player_bonus': sender_bonus,
#                     'participation_fee': Constants.real_participation_fee,
#                     'fee_for_waiting': round(sender.participant.vars['payment_for_wait'], 2),
#                     }
class GameOver(Page):
    """
    This page will be displayed after the last round is over - the experiment is finish.
    It will display the results: the payoff of each player
    """
    template_name = 'text_exp_bot/GameOver.html'

    def is_displayed(self):
        # show this page only after the last round
        if not self.player.participant.vars.get('go_to_the_end', False):  # players who didn't get a partner
            return not self.group.instruction_timeout and not self.group.failed_intro_test and \
                   self.round_number == Constants.num_rounds
        else:
            return False

    def vars_for_template(self):
        # get the number of points of each player and convert to real money
        # receiver total points
        receiver_total_points = sum([p.payoff for p in self.player.in_all_rounds()]) + \
                                self.session.vars['initial_points_receiver']
        #print('receiver_total_points', receiver_total_points)
        receiver_p_to_bonus = float(receiver_total_points / self.player.participant.vars['max_points'])
        #print('receiver_p_to_bonus:', receiver_p_to_bonus)
        receiver_bonus = Constants.bonus if random.random() <= receiver_p_to_bonus else 0
        #print('receiver_bonus', receiver_bonus)
        receiver_total_payoff = receiver_bonus + self.session.config['participation_fee']
        receiver_total_payoff = c(receiver_total_payoff)
        #print('receiver payoff', receiver_total_payoff)

        #print('max points for receiver:', self.player.participant.vars['max_points'])

        self.player.participant.payoff = c(receiver_bonus)
        #print('receiver final payoff:', self.player.participant.payoff)

        return {'player_total_payoff': receiver_total_payoff,
                'player_bonus': self.player.participant.payoff,
                'participation_fee': self.session.config['participation_fee'],}

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
    template_name = 'text_exp_bot/OnePlayerWait.html'

    def is_displayed(self):
        # This page inherits only from Page not from CustomMturkPage: Will appear even to players who have hit the
        # "finish study button" on a CustomMturkWaitPage skip_until_the_end_of
        # show this page also to participants that passed the first test but partners have not
        # show this page also to participants that wait to their partner on instruction page for more than 10 minutes
        #print(f"instruction_timeout: {self.player.participant.vars.get('instruction_timeout')},"
        #      f"failed_intro_test: {self.player.participant.vars.get('failed_intro_test')},"
        #      f"failed_intro_test: {self.group.failed_intro_test},"
        #      f"instruction_timeout: {self.group.instruction_timeout}")
        if self.round_number == 1 and self.player.participant.vars.get('go_to_the_end', False):
            return True
        # elif self.player.participant.vars.get('failed_intro_test'):
        #     return False
        # elif self.player.participant.vars.get('instruction_timeout'):
        #     return False
        else:
            return False
        # return (self.round_number == Constants.num_rounds and self.player.participant.vars.get('go_to_the_end', False))\
        #        or (not self.player.participant.vars.get('failed_intro_test') and self.group.failed_intro_test and not\
        #     self.player.participant.vars.get('instruction_timeout') and self.group.instruction_timeout)

    def vars_for_template(self):
        #print('finish the study')
        if self.player.participant.vars['payment_for_wait'] > Constants.max_payment_for_waiting:
            self.player.participant.vars['payment_for_wait'] = Constants.max_payment_for_waiting
        self.player.participant.payoff = self.player.participant.vars['payment_for_wait']

        return {
            'payment': round(self.player.participant.vars.get('payment_for_wait'), 2)
        }


class AfterAutoSubmit(Page):
    """
    This page will be shown to players that their HIT was automatically submitted
    """
    template_name = 'text_exp_bot/afterAutoSubmit.html'

    def is_displayed(self):
        return (self.round_number == 1 and self.player.participant.vars.get('automate_timeout', False)) or\
               (self.round_number == Constants.num_rounds and
                self.player.participant.vars.get('automate_timeout', False) and self.group.receiver_passed_test == 0)

    def vars_for_template(self):
        #if self.player.participant.vars['payment_for_wait'] > Constants.max_payment_for_waiting:
        #    self.player.participant.vars['payment_for_wait'] = Constants.max_payment_for_waiting

        if self.player.participant.vars.get('failed_intro_test') and self.group.failed_intro_test:
            reason = 'you failed the instruction test'
            reject = True
        #elif not self.player.participant.vars.get('failed_intro_test') and self.group.failed_intro_test and not\
        #        self.player.participant.vars.get('instruction_timeout'):
        #    reason = 'your partner failed the instruction test'
        #    reject = False
        #    self.player.participant.payoff = self.player.participant.vars['payment_for_wait']

        elif self.player.participant.vars.get('instruction_timeout') and self.group.instruction_timeout:
            reason = 'you have not responded on time'
            reject = True
        #elif not self.player.participant.vars.get('instruction_timeout') and self.group.instruction_timeout:
        #    reason = 'your partner has not responded on time'
        #    reject = False
        #    self.player.participant.payoff = self.player.participant.vars['payment_for_wait']

        elif self.player.participant.vars['failed_dm_test']:
            reason = 'you have failed the test'
            reject = True
        else:
            reason = ''
            reject = True
        #self.player.participant.vars['df'].to_csv('oTree_mturk/mcts_/test_df.csv')
        return {
            'reason': reason,
            'reject': reject,
            'payment': 0
        }


page_sequence = [
      #Introduction,
      #IntroTestFeedback,
    #AfterIntroTest,
     #IntroTestFeedback,
     #IntroTimeout,
     #PersonalInformation,
     #AfterInstructions,
     # IntroTimeout,
    #SenderPage,
    ReceiverWaitPage,
    ReceiverPage,
    #ReceiverWaitPage,
    Results,
    Test,
    FeedbackTest,
    GameOver,
    # OnePlayerWait,
    #AfterAutoSubmit
]
