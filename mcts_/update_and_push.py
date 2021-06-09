import pandas as pd
from random import randrange
import torch
# from simple_predictor import *
# from LSTM import train_valid_lstm_text_decision_fix_text_features_model
# from create_save_data_for_simu import emb_for_simulation, bert_and_manual_features
# from crf import LSTMTagger, crf_eval
import random
import joblib
from mcts_.dm_models import crf_eval as evalu
from mcts_.new_emb_try import create_embadding_for_qdn
import numpy as np
from scipy.stats import bernoulli

random.seed(130)
#manual_features_file = joblib.load('/home/mayatarno/PycharmProjects/hotels_interaction/data/verbal/manual_features.pkl')
#bert_features_file = joblib.load('/home/mayatarno/PycharmProjects/hotels_interaction/data/verbal/bert_embedding.pkl')


#
#
# def create_random_choices(interaction, rounds, interaction_id):
#     rounds_to_complete = [i for i in range(1, 11) if i >= rounds]
#     for round_ in rounds_to_complete:  # assume that round-1 is the index of the wanted round
#         action = int(randrange(7))
#         this_round_lottery = interaction.loc[round_ - 1][
#             ['group_score_0', 'group_score_1', 'group_score_2', 'group_score_3', 'group_score_4', 'group_score_5',
#              'group_score_6']].values[randrange(7)]
#         interaction = update_this_round_columns(interaction=interaction, this_round_lottery=this_round_lottery,
#                                                 round_=round_, this_round_payoff=0, action=action,
#                                                 interaction_id=interaction_id)
#         interaction = update_next_round_prev_columns_for_first_round(interaction, round_)
#         if round_ < 10:
#             interaction = update_next_round_prev_columns(interaction=interaction, this_round_lottery=this_round_lottery,
#                                                          round_=round_, this_round_payoff=0)
#     return interaction
#

def find_review_index_given_review(hotel,action,review):
    reviews_features = pd.read_csv('mcts_/hotels_index_test_data.csv')
    #print(reviews_features['hotel'].unique(),hotel)
    if review[0]=='P':
        return reviews_features[(reviews_features.hotel==hotel) & (reviews_features.rev_index==action)&(reviews_features.posorneg== 'Positive')]['review_id'].values[0]
    else:
        return reviews_features[(reviews_features.hotel==hotel) & (reviews_features.rev_index==action)&(reviews_features.posorneg== 'Negative')]['review_id'].values[0]

    # if len(reviews_features[reviews_features['review'] == review]['review_id'].values) == 1:
    #     index = reviews_features[reviews_features['review'] == review]['review_id'].values[0]
    # else:
    #     if review[0] == 'P':
    #         positive_patr, negtive_part = review.split('Negative: ', 1)
    #         new_rev = 'Negative: ' + negtive_part + ' ' + positive_patr
    #         new_rev = new_rev[:-1]
    #     elif review[0] == 'N':
    #         positive_patr, negtive_part = review.split('Positive: ', 1)
    #         new_rev = 'Positive: ' + negtive_part + ' ' + positive_patr
    #         new_rev = new_rev[:-1]
    #     index = reviews_features[reviews_features['review'] == new_rev]['review_id'].values[0]
    # return index


def push_combi_to_interaction(interaction, rounds, action, interaction_id, this_round_payoff):
    #print(interaction[['group_score_0', 'group_score_1', 'group_score_2', 'group_score_3', 'group_score_4', 'group_score_5',
    #     'group_score_6']])
    this_round_lottery = interaction.loc[rounds - 1][
        ['group_score_0', 'group_score_1', 'group_score_2', 'group_score_3', 'group_score_4', 'group_score_5',
         'group_score_6']].values[randrange(7)]
    #print('nn',interaction.loc[rounds - 1][
       # ['group_score_0', 'group_score_1', 'group_score_2', 'group_score_3', 'group_score_4', 'group_score_5',
       #  'group_score_6']])
    interaction = update_this_round_columns(interaction=interaction, this_round_lottery=this_round_lottery,
                                            round_=rounds, this_round_payoff=this_round_payoff, action=action,
                                            interaction_id=interaction_id)
    if rounds < 10:
        interaction = update_next_round_prev_columns(interaction=interaction, this_round_lottery=this_round_lottery,
                                                     round_=rounds, this_round_payoff=this_round_payoff)
    return interaction

#
# def interaction_embadding(interaction, interaction_id, manual_features_file, bert_features_file):
#     interaction = interaction[['pair_id', 'subsession_round_number', 'group_sender_answer_reviews',
#                                'group_lottery_result', 'review_id', 'previous_round_lottery_result',
#                                'previous_round_decision', 'previous_review_id', 'group_average_score',
#                                'lottery_result_low', 'lottery_result_med1', 'previous_round_lottery_result_low',
#                                'previous_round_lottery_result_high', 'previous_average_score_low',
#                                'previous_average_score_high', 'previous_round_lottery_result_med1',
#                                'group_sender_payoff', 'lottery_result_high',
#                                'chose_lose', 'chose_earn', 'not_chose_lose', 'not_chose_earn',
#                                'group_sender_answer_scores', '10_result', 'status', 'player_id_in_group',
#                                'group_receiver_choice', 'previous_score',
#                                'group_random_positive_negative_review_0', 'group_random_positive_negative_review_1',
#                                'group_random_positive_negative_review_2', 'group_random_positive_negative_review_3',
#                                'group_random_positive_negative_review_4', 'group_random_positive_negative_review_5',
#                                'group_random_positive_negative_review_6',
#                                'group_score_0', 'group_score_1', 'group_score_2', 'group_score_3', 'group_score_4',
#                                'group_score_5', 'group_score_6']]
#     # interaction.to_csv('/home/mayatarno/PycharmProjects/hotels_interaction/data/verbal/simu_test.csv')
#     return bert_and_manual_features(interaction, interaction_id, manual_features_file, bert_features_file)

#
# def vanilla_simulation(hotel_order, rounds):
#     next_three_hotels = hotel_order[rounds - 1:rounds + 2]
#     reviews_combination = []
#     if len(next_three_hotels) == 3:
#         for first_hotel in range(0, 7):
#             for second_hotel in range(0, 7):
#                 # for third_hotel in range(0,7):
#                 reviews_combination.append([first_hotel, second_hotel])  # ,third_hotel])
#     elif len(next_three_hotels) == 2:
#         for first_hotel in range(0, 7):
#             for second_hotel in range(0, 7):
#                 reviews_combination.append([first_hotel, second_hotel])
#     elif len(next_three_hotels) == 1:
#         for first_hotel in range(0, 7):
#             reviews_combination.append([first_hotel])
#     else:
#         print('error!!!')
#     return reviews_combination
#
#
# def choose_argmax_action_given_dict(pred_tot):
#     new_tot = {}
#     counter_for_action = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
#     for key, val in pred_tot.items():
#         if len(str(key)) == 1:
#             if key not in new_tot:
#                 new_tot[key] = val
#             else:
#                 counter_for_action[key] += 1
#                 new_tot[key] += val
#         else:
#             if key[0] not in new_tot:
#                 new_tot[key[0]] = val
#             else:
#                 counter_for_action[key[0]] += 1
#                 new_tot[key[0]] += val
#     for key, val in new_tot.items():
#         new_tot[key] = val / counter_for_action[key]
#     max_value = max(new_tot.values())
#     max_keys = [k for k, v in new_tot.items() if v == max_value]
#     return max_keys[0]


def take_an_action(action, rounds, interaction, interaction_id, model,manual_features_file, bert_features_file):
    this_round_lottery = interaction.loc[rounds - 1][
        ['group_score_0', 'group_score_1', 'group_score_2', 'group_score_3', 'group_score_4', 'group_score_5',
         'group_score_6']].values[randrange(7)]

    #     ['group_score_0', 'group_score_1', 'group_score_2', 'group_score_3', 'group_score_4', 'group_score_5',
    #      'group_score_6']].values[randrange(7)]
    interaction1 = update_this_round_columns(interaction=interaction, this_round_lottery=this_round_lottery,
                                             round_=rounds,
                                             this_round_payoff=0, action=action, interaction_id=interaction_id)

    if rounds < 10:
        interaction1 = update_next_round_prev_columns(interaction=interaction1, this_round_lottery=this_round_lottery,
                                                      round_=rounds, this_round_payoff=0)
    #interaction1 = create_random_choices(interaction1, rounds + 1, interaction_id)
    interaction_embaded = create_embadding_for_qdn(interaction1, manual_features_file, True)
    interaction_embaded['labels'] = interaction_embaded['labels_for_probability']
    del interaction_embaded['labels_for_probability']
    prediction = evalu(interaction_embaded,model).reshape(10)[rounds-1].item()
    #if bernuli == True:
    this_round_payoff = int(bernoulli.rvs(size=1, p=max(0, prediction)))
    #else:
    #    this_round_payoff = (1 if prediction > tresh else 0)  # int(bernoulli.rvs(size=1, p=max(0,prediction[rounds - 1])))
    interaction1 = update_this_round_columns(interaction=interaction1, this_round_lottery=this_round_lottery,
                                             round_=rounds,
                                             this_round_payoff=this_round_payoff, action=action,
                                             interaction_id=interaction_id)

    interaction1 = update_next_round_prev_columns_for_first_round(interaction1, rounds)
    if rounds < 10:
        interaction1 = update_next_round_prev_columns(interaction=interaction1, this_round_lottery=this_round_lottery,
                                                      round_=rounds, this_round_payoff=this_round_payoff)
    return this_round_payoff, interaction1, prediction


# def interaction_simu(model, iterator, cuda_device, rounds, interaction, interaction_id,
#                      reviews_features):  # ,reviews_combination):
#     action = which_action_to_take(rounds, interaction, interaction_id, model, iterator, cuda_device)
#     total_payoff, interaction, _ = take_an_action(action, rounds, interaction, interaction_id, model)
#     return total_payoff, interaction, action
#

def update_next_round_prev_columns_for_first_round(interaction, round_):
    interaction.at[round_, 'previous_not_chose_lose'] = np.nan
    interaction.at[round_, 'previous_chose_earn'] = np.nan
    interaction.at[round_, 'previous_not_chose_earn'] = np.nan
    interaction.at[round_, 'previous_round_lottery_result'] = np.nan
    interaction.at[round_, 'previous_round_decision'] = np.nan
    interaction.at[round_, 'previous_round_lottery_result_low'] = np.nan
    interaction.at[round_, 'previous_round_lottery_result_high'] = np.nan
    interaction.at[round_, 'previous_average_score_low'] = np.nan
    interaction.at[round_, 'previous_average_score_high'] = np.nan
    # interaction.at[round_, 'previous_round_lottery_result_med1'] = np.nan
    interaction.at[round_, 'previous_score'] = np.nan
    interaction.at[round_, 'previous_chose_lose'] = np.nan
    return interaction


def update_next_round_prev_columns(interaction, this_round_lottery, round_, this_round_payoff):
    interaction.at[round_, 'previous_not_chose_lose'] = interaction.loc[round_ - 1]['not_chose_lose']
    interaction.at[round_, 'previous_chose_earn'] = interaction.loc[round_ - 1]['chose_earn']
    interaction.at[round_, 'previous_not_chose_earn'] = interaction.loc[round_ - 1]['not_chose_earn']
    interaction.at[round_, 'previous_round_lottery_result'] = this_round_lottery
    interaction.at[round_, 'previous_round_decision'] = this_round_payoff
    interaction.at[round_, 'previous_round_lottery_result_low'] = interaction.loc[round_ - 1]['lottery_result_low']
    interaction.at[round_, 'previous_round_lottery_result_high'] = interaction.loc[round_ - 1]['lottery_result_high']
    interaction.at[round_, 'previous_average_score_low'] = interaction.loc[round_ - 1]['average_score_low']
    interaction.at[round_, 'previous_average_score_high'] = interaction.loc[round_ - 1]['average_score_high']
    interaction.at[round_, 'previous_round_lottery_result_med1'] = interaction.loc[round_ - 1]['lottery_result_med1']
    interaction.at[round_, 'previous_score'] = interaction.loc[round_ - 1]['group_sender_answer_scores']
    interaction.at[round_, 'previous_chose_lose'] = interaction.loc[round_ - 1]['chose_lose']
    return interaction


def update_this_round_columns(interaction, this_round_lottery, round_, this_round_payoff, action, interaction_id):
    #print('lottery:',this_round_lottery)
    review = interaction.at[round_ - 1, f'group_random_positive_negative_review_{int(action)}']
    index = find_review_index_given_review(interaction.at[round_ - 1,'group_average_score'],action,review)
    group_sender_answer_index = action + 1
    interaction.at[round_ - 1, '10_result'] = (1 if int(this_round_lottery) == int(10) else 0)
    interaction.at[round_ - 1, 'group_sender_answer_index'] = group_sender_answer_index
    interaction.at[round_ - 1, 'group_sender_answer_reviews'] = review
    interaction.at[round_ - 1, 'review_id'] = index
    interaction.at[round_ - 1, 'group_sender_answer_scores'] = float(
        interaction.at[round_ - 1, f'group_score_{str(action)}'])
    interaction.at[round_ - 1, 'group_sender_answer_negative_reviews'] = \
        interaction.at[round_ - 1, f'group_negative_review_{str(action)}']
    interaction.at[round_ - 1, 'group_sender_answer_positive_reviews'] = \
        interaction.at[round_ - 1, f'group_positive_review_{str(action)}']
    interaction.at[round_ - 1, 'group_lottery_result'] = this_round_lottery
    interaction.at[round_ - 1, 'group_sender_answer_scores'] = interaction.loc[round_ - 1][f'group_score_{action}']
    interaction.at[round_ - 1, 'lottery_result_low'] = np.where(interaction.group_lottery_result < 3, 1, 0)[round_ - 1]
    interaction.at[round_ - 1, 'lottery_result_med1'] = np.where(interaction.group_lottery_result.between(3, 5), 1, 0)[
        round_ - 1]
    interaction.at[round_ - 1, 'lottery_result_high'] = np.where(interaction.group_lottery_result >= 8, 1, 0)[
        round_ - 1]
    interaction.at[round_ - 1, 'lottery_result_lose'] = np.where(interaction.group_lottery_result < 8, 1, 0)[round_ - 1]
    interaction.at[round_ - 1, 'average_score_low'] = np.where(interaction.group_average_score < 5, 1, 0)[round_ - 1]
    interaction.at[round_ - 1, 'average_score_high'] = np.where(interaction.group_average_score >= 8, 1, 0)[round_ - 1]
    interaction.at[round_ - 1, 'group_sender_payoff'] = this_round_payoff
    interaction.at[round_ - 1, 'chose_lose'] = interaction.loc[round_ - 1]['lottery_result_lose'] * \
                                               interaction.loc[round_ - 1]['group_sender_payoff']
    interaction.at[round_ - 1, 'chose_earn'] = interaction.loc[round_ - 1]['lottery_result_high'] * \
                                               interaction.loc[round_ - 1]['group_sender_payoff']
    interaction.at[round_ - 1, 'group_receiver_choice'] = 1 - this_round_payoff
    interaction.at[round_ - 1, 'not_chose_lose'] = interaction.loc[round_ - 1]['lottery_result_lose'] * \
                                                   interaction.loc[round_ - 1]['group_receiver_choice']
    interaction.at[round_ - 1, 'not_chose_earn'] = interaction.loc[round_ - 1]['lottery_result_high'] * \
                                                   interaction.loc[round_ - 1]['group_receiver_choice']
    interaction.at[round_ - 1, 'group_receiver_payoff'] = float(
        [this_round_lottery - 8 if this_round_payoff == 1 else 0][0])
    interaction.at[round_ - 1, 'dm_expected_payoff'] = float(
        [interaction.loc[round_ - 1]['group_average_score'] - 8 if this_round_payoff == 1 else 0][0])
    return interaction


def update_inretaction(interaction, action, round_, this_round_payoff, interaction_id):
    this_round_lottery = interaction.loc[round_ - 1][
        ['group_score_0', 'group_score_1', 'group_score_2', 'group_score_3', 'group_score_4', 'group_score_5',
         'group_score_6']].values[randrange(7)]
    # this_round_lottery = lottery
    interaction = update_this_round_columns(interaction=interaction, this_round_lottery=this_round_lottery,
                                            round_=round_, this_round_payoff=this_round_payoff, action=action,
                                            interaction_id=interaction_id)
    if round_ < 10:
        interaction = update_next_round_prev_columns(interaction, this_round_lottery, round_, this_round_payoff)
    return interaction
