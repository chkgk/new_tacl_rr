from __future__ import division
from copy import deepcopy
from mcts_.my_mcts import mcts
from functools import reduce
import numpy as np
import torch
from mcts_.update_and_push import update_inretaction,push_combi_to_interaction,take_an_action#,take_an_action_with_bert,take_an_action_with_svm
from mcts_.dm_model_with_bert import bert_eval_mcts
from mcts_.create_bert_embadding import load_reviews_data_mcts
from mcts_.dm_model_with_bert import BertTagger_mcts
import pandas as pd
from scipy.stats import bernoulli
import time
import joblib
from mcts_.new_emb_try import create_embadding_for_qdn
from mcts_.dm_models import LSTMTagger as LSTMDM
from mcts_.dm_models import crf_eval as evalu
from mcts_.Q_model import LSTMTagger as LSTMQ
from scipy.special import softmax
import random
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
manual_features_file =joblib.load('manual_binary_features_test_data.pkl')
bert_features_file = None#joblib.load('bert_embedding_test_data.pkl')
from sklearn.metrics import pairwise
from mcts_.update_and_push import find_review_index_given_review

def personalized(interaction,current_round):
    if current_round==1:
        return random.choice([0,1,2,3,4,5,6])
    interaction1=interaction[(interaction.index<current_round-1)&(interaction.group_sender_payoff==1)]
    rev_features1=pd.read_csv('manual_binary_features_test_data.csv')
    rev_features1['del'] = rev_features1.apply(lambda x: 1 if x['review_id'] not in interaction1['review_id'].values else 0, axis=1)
    rev_features = rev_features1[rev_features1['del']!=1]
    del rev_features['del']
    del rev_features['Unnamed: 41']
    del rev_features1['del']
    del rev_features1['Unnamed: 41']
    avg_f,avg_f1,max_sim=[],[],0
    for col in rev_features.columns:
        if col not in ['review_id','review']:
            avg_f.append(rev_features[col].mean())
    for i in range(7):
        review = interaction[(interaction.index==current_round-1)][f'group_random_positive_negative_review_{i}'].values[0]
        review_id = find_review_index_given_review(interaction[(interaction.index==current_round-1)]['group_average_score'].values[0], i, review)
        review = rev_features1[rev_features1['review_id']==review_id]
        for col in review.columns:
            if col not in ['review_id','review']:
                avg_f1.append(review[col].mean())
        try:
            sim = pairwise.cosine_similarity([np.array(avg_f)],[np.array(avg_f1)])[0][0]
        except:
            sim=0
        if sim>=max_sim:
            max_sim=sim
            action = i
        avg_f1=[]
    return action


def personalized_bert(interaction,current_round):
    if current_round==1:
        return random.choice([0,1,2,3,4,5,6])
    interaction1=interaction[(interaction.index<current_round-1)&(interaction.group_sender_payoff==1)]
    rev_features1=joblib.load('bert_embedding_test_data.pkl')
    rev_features1['del'] = rev_features1.apply(lambda x: 1 if x['review_id'] not in interaction1['review_id'].values else 0, axis=1)
    rev_features = rev_features1[rev_features1['del']!=1]
    del rev_features['del']
    #del rev_features['Unnamed: 41']
    del rev_features1['del']
    #del rev_features1['Unnamed: 41']
    avg_f,avg_f1,max_sim=[],[],0
    try:
        avg_f=np.array(rev_features['review_features'].values).mean()
    except:
        return 3
    for i in range(7):
        review = interaction[(interaction.index==current_round-1)][f'group_random_positive_negative_review_{i}'].values[0]
        review_id = find_review_index_given_review(interaction[(interaction.index==current_round-1)]['group_average_score'].values[0], i, review)
        review = rev_features1[rev_features1['review_id']==review_id]
        avg_f1=np.array(review['review_features'].values).mean()
        try:
            sim = pairwise.cosine_similarity([np.array(avg_f)],[np.array(avg_f1)])[0][0]
        except:
            sim=0
        if sim>=max_sim:
            max_sim=sim
            action = i
        avg_f1=[]
    return action


def kb_baisline(state,i):
    hotels_dic = {9.66: [4, 5, 6],  #
                  9.71: [4, 5, 6],  #
                  9.04: [5, 3, 6, 4],  #
                  7.74: [4, 6, 5, 3, 2],
                  4.56: [2, 5, 3, 4, 6, 1, 0],  #
                  8.33: [3, 2, 5, 4, 6],  #
                  6.69: [2, 5, 3, 4, 6, 1, 0],
                  7.91: [4, 3, 5, 6, 2],
                  8.04: [4, 5, 3, 6, 2, 1],  #
                  8.89: [4, 6, 5]}  #
    return random.choice(hotels_dic[state.at[i - 1, 'group_average_score']])

def tacl(rejectness):
    if rejectness==0:
        return random.choice([6])
    elif rejectness==1:
        return random.choice([4,5])
    else:
        return 3

def last_dec_(last_dec):
    if last_dec==1:
        return random.choice([3,4,5,6])
    else:
        return random.choice([0,1,2])



def greedy_baisline(initialState, value_model,type,epsilon):
    Qsa = {}
    for possible_action in range(0, 7):  # [0,1,2..6]
        demo_state = push_combi_to_interaction(interaction=initialState.board, rounds=initialState.currentRound,
                                               action=possible_action, interaction_id=initialState.interaction_id,
                                               this_round_payoff=0)
        demo_state_embadded = create_embadding_for_qdn(demo_state[demo_state.index <= initialState.currentRound - 1],
                                                       manual_features_file, True)
        Qsa[possible_action] = round(evalu(demo_state_embadded[[f'features_round_{i}' for i in range(1,initialState.currentRound+1) ]],
                                          value_model, 'Q')[0][-1].item())

    if type=='hard':
        if random.uniform(0,1)<=epsilon:
            return random.choice([0,1,2,3,4,5,6])
        else:
            return random.choice([k for k, v in Qsa.items() if v == max(Qsa.values())])
    return list(np.random.multinomial(1, softmax(np.array(list(Qsa.values()))), size=1)[0]).index(1)#[k for k, v in Qsa.items() if v == max_value][0]




def greedy_baisline_dm_model(initialState, dm_model, type):
    Qsa = {}
    for possible_action in range(0, 7):  # [0,1,2..6]
        demo_state = push_combi_to_interaction(interaction=initialState.board, rounds=initialState.currentRound,
                                               action=possible_action, interaction_id=initialState.interaction_id,
                                               this_round_payoff=0)
        demo_state_embadded = create_embadding_for_qdn(demo_state,
                                                       manual_features_file, True)
        Qsa[possible_action] = evalu(demo_state_embadded[[f'features_round_{initialState.currentRound}', 'labels']],
                                          dm_model, 'dm').item()
    #if demo_state[(demo_state.index == initialState.currentRound - 1)]['group_average_score'].values[0]==9.66:
    #    print(Qsa)
    #max_value = max(Qsa.values())
    if type=='hard':
        return [k for k, v in Qsa.items() if v == max(Qsa.values())][0]
    return list(np.random.multinomial(1, softmax(np.array(list(Qsa.values()))), size=1)[0]).index(1)#[k for k, v in Qsa.items() if v == max_value][0]



#for val in list(manual_features_file.columns):
#    if val not in ['review_features','review_id', 'review','score']:
#        del manual_features_file[val]
#joblib.dump(manual_features_file,'/home/mayatarno/PycharmProjects/hotels_interaction/data/verbal/manual_features.pkl')

class NaughtsAndCrossesState():
    def __init__(self, interaction, interaction_id, model,currentRound,exp_tot_payoff,treshold,with_bert=None):
        self.board = interaction
        self.currentPlayer = 1
        self.last_rev_ind = 0
        self.initial_round = currentRound
        self.currentRound = currentRound
        self.interaction_id = interaction_id
        self.treshold =treshold
        self.model = model
        self.exp_tot_payoff = exp_tot_payoff
        self.isTerminal_ = False
        self.with_bert=with_bert
    def getCurrentPlayer(self):
        return self.currentPlayer

    def getPossibleActions(self):
        if self.currentPlayer==1:
            return[0,1,2,3,4,5,6]#most_common_reviews_per_hotel_by_human_users(self.board[self.board['subsession_round_number']==self.currentRound]['group_average_score'].values[0])

    def getfeaturesfile(self):
        return manual_features_file, bert_features_file

    def takeAction(self, action):
        newState = deepcopy(self)
        if self.currentPlayer == 1:
            # self.board[self.currentRound]['expert_review'] = action
            newState.board = push_combi_to_interaction(interaction=self.board, rounds=self.currentRound, action=action,
                                                       interaction_id=self.interaction_id, this_round_payoff=0)
            # print(f"push_combi_to_interaction--- %s seconds ---" % float(time.time() - start_time))
            newState.exp_tot_payoff = self.exp_tot_payoff
            newState.currentRound = self.currentRound
            newState.last_rev_ind = action
            # interaction_embaded = create_embadding_for_qdn(newState.board, joblib.load('new_bert_embedding_test_data.pkl'), True)
            # interaction_embaded = interaction_embaded[f'features_round_{newState.currentRound}']
            # import pickle
            # prediction = self.model.predict(list(interaction_embaded))[0]

            interaction_embaded = create_embadding_for_qdn(newState.board[(newState.board.index<newState.currentRound)], manual_features_file,
                                                               True)
            prediction = \
            evalu(interaction_embaded[[f'features_round_{i}' for i in range(1, newState.currentRound + 1)]],
                      self.model)[0][-1].item()
            receiver_action = int(1 if prediction > self.treshold else 0)
            newState.exp_tot_payoff += receiver_action
            newState.board = update_inretaction(interaction=self.board, action=int(newState.last_rev_ind),
                                                round_=newState.currentRound, this_round_payoff=receiver_action,
                                                interaction_id=self.interaction_id)
        newState.model = self.model
        newState.interaction_id = self.interaction_id
        newState.isTerminal_ = self.isTerminal_
        if newState.currentRound == 10:
            newState.isTerminal_ = True
        else:
            newState.currentRound += 1
        return newState

    def isTerminal(self):
        if  (self.currentRound == 10 and self.isTerminal_==True) or self.currentRound - self.initial_round == 10:#self.currentRound - self.initial_round == 3 or
            return True
        else:
            return False

    def getReward(self):
        return self.exp_tot_payoff


class Action():
    def __init__(self, player):
        self.player = player


    def __str__(self):
        return str((self.x, self.y))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.x == other.x and self.y == other.y and self.player == other.player

    def __hash__(self):
        return hash((self.x, self.y, self.player))


def mcts_live_simu(all_interaction,round):
    dm_model_name = 'mcts_/0.824_20_behavioral_features+text_all_data_epoch_100_batch_5_hid_dim_256_drop_0.3.th'  # all_data_sigmoid_new_with_manual_updated_epoch_62_batch_20_hid_dim_100.th'#transformer_fscore_0.6552655759833379_acc_0.8458333333333333_epoch_98_batch_5_hid_dim_100_fold_'
    q_model_name = 'mcts_/0.389_20_features_behavioral+textual_all_data_epoch_100_batch_10_hid_dim_128_drop_0.4.th'  # all_data_sigmoid_new_with_manual_updated_epoch_62_batch_20_hid_dim_100.th'#transformer_fscore_0.6552655759833379_acc_0.8458333333333333_epoch_98_batch_5_hid_dim_100_fold_'
    treshold = 0.5
    model = LSTMDM(embedding_dim=59, hidden_dim=256, tagset_size=1, dropout=0.3)
    value_model = LSTMQ(embedding_dim=59, hidden_dim=128, tagset_size=1, dropout=0.4)
    model.load_state_dict(
        (torch.load(dm_model_name,map_location='cpu')))  # torch.load('model_0.708433734939759_0.8319444444444445_50_1606.pkl')
    model.eval()
    value_model.load_state_dict(torch.load(q_model_name,map_location='cpu'))
    value_model = value_model.to(device)
    value_model.eval()
    timeLimit = 90000#10000#
    tot_pay, count = 0, 0
    interaction = all_interaction.reset_index()
    currentround = round
    initialState = NaughtsAndCrossesState(interaction_id=1, model=model, interaction=interaction,
                                                      currentRound=currentround, exp_tot_payoff=sum(interaction[interaction.subsession_round_number<currentround]['group_sender_payoff'].values), treshold=treshold)
    mcts1 = mcts(timeLimit=timeLimit, explorationConstant=0.5)
    start_time=time.time()
    print('mcts is begining - 21/5')
    action, _ = mcts1.search(initialState=initialState, value_model=value_model,
                                      manual_features_file=manual_features_file, bert_features_file=bert_features_file)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(f'mcts returns action {action}!!!!!')
    return action
