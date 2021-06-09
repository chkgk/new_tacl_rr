import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import joblib
import random
import numpy as np
import copy
from sklearn import metrics
import time
from mcts_.data_preperation_functions import cross_vali
###############
#batch_size = 5
epoch_num = 100
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_type = 'blogic'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
    #else:
    torch.cuda.manual_seed(seed)

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size,dropout):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,dropout=dropout)#, dropout=dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.drop = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sentence_emb):
        others, text = sentence_emb[:, :, :20], self.sigmoid(sentence_emb[:, :, 20:])
        sentence_emb = torch.cat([others, text], dim=2)
        lstm_out, _ = self.lstm(sentence_emb)
        lstm_out = self.drop(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space
class NOLSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size,dropout):
        super(NOLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,dropout=dropout)#, dropout=dropout)
        self.fc = nn.Linear(embedding_dim, tagset_size)
        self.drop = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sentence_emb):
        others, text = sentence_emb[:, :, :20], self.sigmoid(sentence_emb[:, :, 20:])
        sentence_emb = torch.cat([others, text], dim=2)
        #lstm_out, _ = self.lstm(sentence_emb)
        #lstm_out = self.drop(lstm_out)
        tag_space = self.fc(sentence_emb)
        return tag_space

def crete_random_batches(train_df, batch_size,training_data,shuflle_batch):
    if shuflle_batch==False:
        return training_data
    training_data = []
    count = 0
    batch=[]
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    for index, row in train_df.iterrows():
        features = [row[col] for col in list(train_df.columns) if 'features_round' in col]
        labels = row['labels']#[np.int_(1) if val == 'hotel' else np.int_(0) for val in row['labels']]
        batch.append((features, labels))
        count += 1
        if count == batch_size or index==len(train_df)-1:
            training_data.append(batch)
            batch = []
            count = 0
    return training_data


def training(fold,data,batch_size,hidden_dim,treshold,dropout,shuflle_batch,results_df, new_index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = nn.BCEWithLogitsLoss()#pos_weight=torch.tensor([pos_weight]).to(device))  # pos_weight=torch.tensor([0.9]))#MSELoss()#

    embedding_dim = len(data.loc[0]['features_round_1'])#+1
    print(embedding_dim)
    model = NOLSTMTagger(embedding_dim=embedding_dim, hidden_dim=hidden_dim, tagset_size=1,dropout=dropout)
    model = model.to(device)
    optimizer =  torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
    count = 0
    training_data, testing_data, batch = [], [], []
    batch1 = []
    pairs_train = []
    train_df = data[data['fold']!=fold]
    test = data[data['fold']==fold]
    for index, row in train_df.iterrows():
        features = [row[col] for col in list(train_df.columns) if 'features_round' in col]
        labels = row['labels']#[np.int_(1) if val == 'hotel' else np.int_(0) for val in row['labels']]
        batch.append((features, labels))
        count += 1
        if count == batch_size or index==len(train_df):
            training_data.append(batch)
            batch = []
            count = 0
            pairs_train.append(data.at[index,'pair_id'])
    for index,row in test.iterrows():
            features = [row[col] for col in list(test.columns) if 'features_round' in col]
            labels = row['labels']#labels = [np.int_(1) if val == 'hotel' else np.int_(0) for val in row['labels']]
            batch1.append((features, labels))
    testing_data.append(batch1)
    epoch_counter=0
    max_f_score_for_fold ,min_loss= 0,100
    for epoch in range(epoch_num):
        epoch_counter+=1
        training_data = crete_random_batches(train_df, batch_size,training_data,shuflle_batch)
        tot_loss=0
        for batch in training_data:
            sentence_ = [sample[0] for sample in batch]
            tags_ = [sample[1] for sample in batch]
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            sentence = torch.Tensor(sentence_).to(device)
            model.zero_grad()
            tag_scores = model(sentence)  # .view(len(sentence), 1, -1))

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            real_teg = torch.Tensor(tags_).to(device)
            if loss_type == 'nll':
                loss = loss_function(nn.LogSoftmax(dim=1)(tag_scores.reshape(real_teg.shape[0], 2, 10)),
                                     real_teg.long())
            elif loss_type == 'blogic':
                loss = loss_function(tag_scores.reshape(real_teg.shape[0],10), real_teg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            tot_loss+=loss.item()
        #print(tot_loss/len(training_data))
        if epoch_counter>=10:
            model.eval()
            with torch.no_grad():
                for batch in testing_data:
                    sentence_ = [sample[0] for sample in batch]
                    tags_ = [sample[1] for sample in batch]
                    # Step 1. Remember that Pytorch accumulates gradients.
                    # We need to clear them out before each instance
                    sentence = torch.Tensor(sentence_).to(device)
                    tag_scores1 = model(sentence)  # .view(len(sentence), 1, -1))
                    tag_scores = torch.sigmoid(tag_scores1)
                    # Step 4. Compute the loss, gradients, and update the parameters by
                    #  calling optimizer.step()
                    real_teg = torch.Tensor(tags_).to(device)
                    #for treshold_ in treshold:
                    y, pred = list(np.array(real_teg.reshape(real_teg.shape[0] * real_teg.shape[1]).cpu())),[1 if val>=0.5 else 0 for val in list(np.array(tag_scores.reshape(real_teg.shape[0] * real_teg.shape[1]).cpu()))]
                    loss = loss_function(tag_scores1.reshape(real_teg.shape[0], 10), real_teg)

                    f_score_home = metrics.f1_score(y, pred,pos_label=0)
                    acc = metrics.accuracy_score(y, pred)
                    #    print(fold, loss.item())
                        #if f_score_home>max_f_score_for_fold:
                    if min_loss>loss.item():
                        min_loss = loss.item()
                        min_loss_model = copy.deepcopy(model).state_dict()
                        max_epoch = epoch_counter
                    results_df.loc[new_index]=[epoch_counter,batch_size,hidden_dim,0.5,dropout,shuflle_batch,f_score_home,acc,fold, copy.deepcopy(model).state_dict(),loss.item()]
                    new_index+=1
                        #if treshold_==0.5:
                        #print(f_score_home,acc)

            model.train()
    return results_df,new_index




def training_all(data,batch_size,hidden_dim,dropout,stop_at,shuflle_batch):
    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = nn.BCEWithLogitsLoss()#pos_weight=torch.tensor([pos_weight]).to(device))  # pos_weight=torch.tensor([0.9]))#MSELoss()#

    embedding_dim = len(data.loc[0]['features_round_1'])#+1
    print(embedding_dim)
    model = LSTMTagger(embedding_dim=embedding_dim, hidden_dim=hidden_dim, tagset_size=1,dropout=dropout)
    model = model.to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)#optim.Adam(model.parameters(), lr=0.01)
    count = 0
    training_data, testing_data, batch = [], [], []
    batch1 = []
    pairs_train = []
    train_df = data#[data['fold']!=fold]
    for index, row in train_df.iterrows():
        features = [row[col] for col in list(train_df.columns) if 'features_round' in col]
        labels = row['labels']#[np.int_(1) if val == 'hotel' else np.int_(0) for val in row['labels']]
        batch.append((features, labels))
        count += 1
        if count == batch_size or index==len(train_df):
            training_data.append(batch)
            batch = []
            count = 0
            pairs_train.append(data.at[index,'pair_id'])
    epoch_counter=0
    for epoch in range(epoch_num):
        epoch_counter+=1
        training_data = crete_random_batches(train_df, batch_size,training_data,shuflle_batch)
        for batch in training_data:
            sentence_ = [sample[0] for sample in batch]
            tags_ = [sample[1] for sample in batch]
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            sentence = torch.Tensor(sentence_).to(device)
            model.zero_grad()
            tag_scores = model(sentence)  # .view(len(sentence), 1, -1))

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            real_teg = torch.Tensor(tags_).to(device)
            if loss_type == 'nll':
                loss = loss_function(nn.LogSoftmax(dim=1)(tag_scores.reshape(real_teg.shape[0], 2, 10)),
                                     real_teg.long())
            elif loss_type == 'blogic':
                loss = loss_function(tag_scores.reshape(real_teg.shape[0],10), real_teg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        if epoch_counter==stop_at:
            torch.save(model.state_dict(), f'dm_model_without_bert/1textual_and_behavioral_game_features_all_data_epoch_{epoch_counter}_batch_{batch_size}_hid_dim_{hidden_dim}_drop_{dropout}.th')
                        #model.state_dict()
            #print(metrics.confusion_matrix(y, pred))





def check_max_cross_val(df_all_thres,tresholds,batch,hid_dim,data,type,f_home_max,acc_max,drop):
    for trsh in tresholds:
        df = df_all_thres[df_all_thres['treshold']==trsh]
        df1 = df[(df.fold == 1) ]
        df2 = df[df['fold'] == 2]
        df3 = df[df['fold'] == 3]
        df4 = df[df['fold'] == 4]
        df5 = df[df['fold'] == 5]
        df1['f_score_home_1'] = df1.apply(lambda x: x['fscore_home'], axis =1)
        df2['f_score_home_2'] =df2.apply(lambda x: x['fscore_home'], axis =1)
        df3['f_score_home_3']=df3.apply(lambda x: x['fscore_home'], axis =1)
        df4['f_score_home_4']=df4.apply(lambda x: x['fscore_home'], axis =1)
        df5['f_score_home_5']=df5.apply(lambda x: x['fscore_home'], axis =1)
        df1['loss_1'] = df1.apply(lambda x: x['loss'], axis =1)
        df2['loss_2'] =df2.apply(lambda x: x['loss'], axis =1)
        df3['loss_3']=df3.apply(lambda x: x['loss'], axis =1)
        df4['loss_4']=df4.apply(lambda x: x['loss'], axis =1)
        df5['loss_5']=df5.apply(lambda x: x['loss'], axis =1)
        df_ = df1.merge(df2, on=['epoch'],how='left')
        df_ = df_.merge(df3,on=['epoch'],how='left')
        df_ = df_.merge(df4, on=['epoch'],how='left')
        df_ = df_.merge(df5, on=['epoch'],how='left')
        df_['avg_f_score_home'] = df_.apply(lambda x: (x['f_score_home_1'] + x['f_score_home_2'] + x['f_score_home_3'] + x[
            'f_score_home_4'] + x['f_score_home_5']) / 5, axis=1)
        df_['avg_loss'] = df_.apply(lambda x: (x['loss_1'] + x['loss_2'] + x['loss_3'] + x[
            'loss_4'] + x['loss_5']) / 5, axis=1)
        #print(max(df_['avg_acc'].values))
        #with open("test_res1.txt", "a") as myfile:
        #    myfile.write(f'acc_{max(df_["avg_acc"].values)}_epoch_{df_[df_["avg_acc"]==max(df_["avg_acc"].values)]["epoch"].values[0]}_batch_{batch}_hid_dim_{hid_dim}_tresh_{trsh}')
        #    myfile.write('\n')
        #print(f'avg_loss: {min(df_["avg_loss"].values)}')
        min_until_now_loss = min(df_['avg_loss'].values)
        max_until_now_fscore = max(df_['avg_f_score_home'].values)
        max_until_now_fscore = copy.deepcopy(df_[df_['avg_f_score_home'] == max_until_now_fscore]['avg_f_score_home'].values[0])
        #if max_until_now_fscore>=0.58:
            #acc = copy.deepcopy(max_until_now_acc)
        epoch = copy.deepcopy(df_[df_['avg_f_score_home']==max_until_now_fscore]['epoch'].values[0])
        max_tre = copy.deepcopy(trsh)
        df = df_all_thres[(df_all_thres.treshold == max_tre) &  (df_all_thres.epoch == epoch)]
        for index,row in df.iterrows():
            torch.save(row['model'], f'NOLSTM_TACL/DM_HC_fscore:{max_until_now_fscore}_text_sig_new_{type}_{max_until_now_fscore}_epoch_{epoch}_batch_{batch}_hid_dim_{hid_dim}_tresh_{max_tre}_drop_{drop}_fold_{row["fold"]}.th')
        joblib.dump(data, f'NOLSTM_TACL/DM_HC_fscore:{max_until_now_fscore}_text_sig_new_{type}_{max_until_now_fscore}_epoch_{epoch}_batch_{batch}_hid_dim_{hid_dim}_tresh_{max_tre}.pkl')
        print('save!!!!!!!!!!!!!!!!!!!!')



def crf_eval(test,model,model_type='dm'):
    model.eval()
    batch = []
    with torch.no_grad():
        for index,row in test.iterrows():
                features = [row[col] for col in list(test.columns) if 'features_round' in col]
                #labels = row['labels']#[np.int_(1) if val == 'hotel' else np.int_(0) for val in row['labels']]
                batch.append((features, []))
        sentence = [sample[0] for sample in batch]
        #sentence = sentence.reshape([1, sentence.shape[0], sentence.shape[1]])
        #model.zero_grad()
        model.to(device)
        #print(sentence)
        sentence = torch.Tensor(sentence).to(device)
        if model_type=='dm':
            tag_scores = torch.sigmoid(model(sentence))#.reshape(1,10).cpu()[0].tolist()  # .view(len(sentence), 1, -1))
        else:
            tag_scores = model(sentence)#.reshape(1,10).cpu()[0].tolist()  # .view(len(sentence), 1, -1))
        return tag_scores


if __name__ == '__main__':
    bath_sizes = [10,15,25,20,5]
    hidden_dims = [64,128,256]#,[50,80,100]
    tresholds = [0.5]#,0.53,0.6,0.52,0.56,0.57]#],0.7,0.75]#[0.63,0.65,0.7,0.75,0.73,0.67,0.77,0.8,0.83,0.5,0.55,0.6,0.53]
    dropouts = [0.4,0.3,0.5,0.6
                ]#[0.5,0.4,0.3,0.2]#[0.55,0.5,0.6,0]
    shuflle_batch = [True]
    import pandas as pd
    cross_df = pd.DataFrame(columns=['param', 'f_home_max', 'acc', 'epoch'])
    new_index = 0
    type ='10.1train_with_expert_behavioral_features'#'10.1train'#'game_and_manual_63_features_train'#'31.12manualtrain'# 'game_and_manual_63_features_train'#'prev_embedding'#'new_with_manual_updated'#'new_with_manual_rext_norm'#new_maya_emb_for_val_new_features_with_histoty_ewm
    #data = joblib.load(f'data/verbal/{files_names[0]}.pkl')#new_emb_for_maya_trial_manual_and_bert.pkl' )#+ file_name)
    data = joblib.load(f'embaddings/{type}.pkl')#embaddings/new_maya_emb_for_val_new_features_{type}.pkl')#_with_bert#('1-10_bert_and_manual_avg_history_embadding.pkl')
    del data['labels']
    data['labels'] = data['labels_for_probability']
    del data['labels_for_probability']
    results_df = pd.DataFrame(columns=['epoch', 'batch', 'hid_dim', 'treshold', 'drop', 'shuflle', 'fscore_home', 'acc', 'fold','model','loss'])
    ind = len(cross_df)
    counter = 0
    f_home_max, acc_max = -10000, -10000
    while counter<1:
        for batch in bath_sizes:
            data = cross_vali(data,10)
            for shuflle in shuflle_batch:
                for drop in dropouts:
                    for hid_dim in hidden_dims:
                            #for pos_weight in pos_weights:
                        set_seed(111)#67
                        start_time = time.time()
                        results_df = pd.DataFrame(columns=[ 'epoch', 'batch', 'hid_dim', 'treshold', 'drop', 'shuflle','fscore_home', 'acc', 'fold','model','loss'])
                        new_index = 0
                        #training_all(data, 5,256, 0.3, 100,shuflle)
                        for fold in range(1,6):
                            results_df, new_index = training(fold,data, batch, hid_dim, tresholds, drop, shuflle,
                                                             results_df, new_index)
                        check_max_cross_val(results_df,tresholds,batch,hid_dim,data,type,f_home_max,acc_max,drop)
        counter+=1
