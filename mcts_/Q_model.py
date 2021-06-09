import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import joblib
import random
import numpy as np
from sklearn import metrics
import time
import copy
from mcts_.data_preperation_functions import cross_vali

###############
#batch_size = 5
epoch_num = 100
#hidden_dim = 100
#dropout = 0.5
#treshold = 0.5

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
    #else:
    torch.cuda.manual_seed(seed)

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_type = 'blogic'

###############
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size,dropout):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.tanh=nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(dropout)
    def forward(self, sentence_emb):
        #sentence_emb = self.drop1(sentence_emb)
        others, text = sentence_emb[:, :, :15], self.sigmoid(sentence_emb[:, :, 15:])
        #emb = emb.reshape(others.shape[0], others.shape[1], others.shape[2])
        sentence_emb = torch.cat([others, text], dim=2)
        lstm_out, _ = self.lstm(sentence_emb)
        lstm_out = self.drop(lstm_out)
        #lstm_out = self.tanh(lstm_out)#, dim=1)
        tag_space = self.hidden2tag(lstm_out)

        return tag_space
class NOLSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size,dropout):
        super(NOLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout)
        self.fc = nn.Linear(embedding_dim, tagset_size)
        self.tanh=nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(dropout)
    def forward(self, sentence_emb):
        #sentence_emb = self.drop1(sentence_emb)
        others, text = sentence_emb[:, :, :20], self.sigmoid(sentence_emb[:, :, 20:])
        #emb = emb.reshape(others.shape[0], others.shape[1], others.shape[2])
        sentence_emb = torch.cat([others, text], dim=2)
        #lstm_out, _ = self.fc(sentence_emb)
        #lstm_out = self.drop(lstm_out)
        #lstm_out = self.tanh(lstm_out)#, dim=1)
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
    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = nn.MSELoss()#pos_weight=torch.tensor([pos_weight]).to(device))  # pos_weight=torch.tensor([0.9]))#MSELoss()#

    embedding_dim = len(data.loc[0]['features_round_1'])
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
        features = [row[col] for col in list(train_df.columns) if 'features_round' in col ]
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
    max_f_score_for_fold = 0
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
            train_loss = loss_function(tag_scores.reshape(real_teg.shape[0],10), real_teg)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        if epoch_counter>=10:
            #print(f'train_loss:{train_loss}')
            model.eval()
            with torch.no_grad():
                for batch in testing_data:
                    sentence_ = [sample[0] for sample in batch]
                    tags_ = [sample[1] for sample in batch]
                    # Step 1. Remember that Pytorch accumulates gradients.
                    # We need to clear them out before each instance
                    sentence = torch.Tensor(sentence_).to(device)
                    tag_scores = model(sentence)  # .view(len(sentence), 1, -1))
                    real_teg = torch.Tensor(tags_).long().to(device)
                    test_loss = loss_function(tag_scores.reshape(real_teg.shape[0], 10), real_teg)
                    f_score_home = test_loss.item()#metrics.f1_score(y, pred,pos_label=0)
                    acc = metrics.accuracy_score(torch.round(tag_scores.reshape(real_teg.shape[0]*10).cpu()),real_teg.reshape(real_teg.shape[0]*10).cpu())
                    #print(f'acc:{acc}')
                    #print(fold)
                    results_df.loc[new_index]=[epoch_counter,batch_size,hidden_dim,dropout,shuflle_batch,f_score_home,acc,fold,  copy.deepcopy(model).state_dict()]
                    new_index+=1
            model.train()
    return results_df,new_index



def training_all(data,batch_size,hidden_dim,dropout,shuflle_batch,stop_at):
    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = nn.MSELoss()#pos_weight=torch.tensor([pos_weight]).to(device))  # pos_weight=torch.tensor([0.9]))#MSELoss()#

    embedding_dim = len(data.loc[0]['features_round_1'])
    model = LSTMTagger(embedding_dim=embedding_dim, hidden_dim=hidden_dim, tagset_size=1,dropout=dropout)
    model = model.to(device)
    optimizer =  torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
    count = 0
    training_data, testing_data, batch = [], [], []
    batch1 = []
    pairs_train = []
    train_df = data
    for index, row in train_df.iterrows():
        features = [row[col] for col in list(train_df.columns) if 'features_round' in col and int(col.split('_')[-1])>3]
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
            train_loss = loss_function(tag_scores.reshape(real_teg.shape[0],10), real_teg)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        if epoch_counter==stop_at:# or epoch_counter==65:
            torch.save(model.state_dict(), f'q_model_without_bert/all_data_epoch_{stop_at}_batch_{batch_size}_hid_dim_{hidden_dim}_drop_{dropout}.th')
            #print(f'special!_all_data_{type}_epoch_{epoch_counter}_batch_{batch_size}_hid_dim_{hidden_dim}.th')



def check_max_cross_val(df,tresholds,batch,hid_dim,data,type,drop):
    max_until_now_fscore = 0
    if True==True:
        df1 = df[df['fold'] == 1]
        df2 = df[df['fold'] == 2]
        df3 = df[df['fold'] == 3]
        df4 = df[df['fold'] == 4]
        df5 = df[df['fold'] == 5]
        df1['loss_1'] = df1.apply(lambda x: x['fscore_home'], axis =1)
        df2['loss_2'] =df2.apply(lambda x: x['fscore_home'], axis =1)
        df3['loss_3']=df3.apply(lambda x: x['fscore_home'], axis =1)
        df4['loss_4']=df4.apply(lambda x: x['fscore_home'], axis =1)
        df5['loss_5']=df5.apply(lambda x: x['fscore_home'], axis =1)
        df1['acc_1'] = df1.apply(lambda x: x['acc'], axis =1)
        df2['acc_2'] =df2.apply(lambda x: x['acc'], axis =1)
        df3['acc_3']=df3.apply(lambda x: x['acc'], axis =1)
        df4['acc_4']=df4.apply(lambda x: x['acc'], axis =1)
        df5['acc_5']=df5.apply(lambda x: x['acc'], axis =1)
        df_ = df1.merge(df2, on=['epoch'])
        df_ = df_.merge(df3, on=['epoch'])
        df_ = df_.merge(df4, on=['epoch'])
        df_ = df_.merge(df5, on=['epoch'])
        df_['avg_loss'] = df_.apply(lambda x: (x['loss_1'] + x['loss_2'] + x['loss_3'] + x['loss_4'] + x['loss_5']) / 5, axis=1)
        df_['avg_acc'] = df_.apply(lambda x: (x['acc_1'] + x['acc_2'] + x['acc_3'] + x[
            'acc_4'] + x['acc_5']) / 5, axis=1)
        # if max(df_['avg_f_score_home'].values)>max_until_now_fscore:
        min_until_now_loss=min(df_['avg_loss'].values)
        acc =df_[df_['avg_loss']==min_until_now_loss]['avg_acc'].values[0]
        epoch = df_[df_['avg_loss']==min_until_now_loss]['epoch'].values[0]
        #with open("test_res_q.txt", "a") as myfile:
        #    myfile.write(f'min_until_now_loss_{min_until_now_loss}_acc_{acc}_epoch_{epoch}_batch_{batch}_hid_dim_{hid_dim}')
        #    myfile.write('\n')
        #     max_tre = 0
        #print(min_until_now_loss,acc)
        if True==True:#min_until_now_loss <= 0.889:# and acc>=0.45:
             df = df[df['epoch'] == epoch]
             for index,row  in df.iterrows():
                 torch.save(row['model'], f'NOLSTM_TACL/Q_HC_loss_{min_until_now_loss}_acc_{acc}_epoch_{epoch}_batch_{batch}_hid_dim_{hid_dim}_drop_{drop}_fold_{row["fold"]}.th')
             joblib.dump(data, f'NOLSTM_TACL/Q_HC_loss_{min_until_now_loss}_acc_{acc}_epoch_{epoch}_batch_{batch}_hid_dim_{hid_dim}_drop_{drop}.pkl')
             print('save!!!!!!!!!!!!!!!!!!!!')
        #return min(min_until_now_loss,min_loss) , acc,epoch


def crf_eval(test,model):
    with torch.no_grad():
        batch1, testing_data  =[], []
        for index,row in test.iterrows():
                features = [row[col] for col in list(test.columns) if 'features_round' in col]
                labels = [np.int_(1) if val == 'hotel' else np.int_(0) for val in row['labels']]
                batch1.append((features, labels))
        testing_data.append(batch1)
        batch = testing_data[0]
        sentence = torch.Tensor(batch[0][0])
        sentence = sentence.reshape([1, sentence.shape[0], sentence.shape[1]])
        model.zero_grad()
        tag_scores = torch.sigmoid(model(sentence)).reshape(1,10).cpu()[0].tolist()  # .view(len(sentence), 1, -1))
        return tag_scores


if __name__ == '__main__':
    #files_names = ['new_emb_for_maya_trial_manual_and_bert']#,'new_emb_for_maya_trial_manual_and_bert.pkl','full_history_amb_plus_bert_norm.pkl']
    bath_sizes = [10,15,25,20,5]
    hidden_dims = [64,128,256]
    tresholds = [0.5]#,0.67,0.7,0.75]#[0.63,0.65,0.7,0.75,0.73,0.67,0.77,0.8,0.83,0.5,0.55,0.6,0.53]
    dropouts = [0.4,0.5,0.6]
    #pos_weights = [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
    shuflle_batch = [True]
    import pandas as pd

    cross_df = pd.DataFrame(columns=['param', 'f_home_max', 'acc', 'epoch'])
    new_index = 0
    type = '10.1train_with_expert_behavioral_features'  # 'prev_embedding'#'new_with_manual_updated'#'new_with_manual_rext_norm'#new_maya_emb_for_val_new_features_with_histoty_ewm
    # data = joblib.load(f'data/verbal/{files_names[0]}.pkl')#new_emb_for_maya_trial_manual_and_bert.pkl' )#+ file_name)
    data = joblib.load(f'embaddings/{type}.pkl')#embaddings/new_maya_emb_for_val_new_features_{type}.pkl')#_with_bert#('1-10_bert_and_manual_avg_history_embadding.pkl')
    del data['labels_for_probability']
    #results_df= pd.DataFrame(columns=['epoch', 'batch', 'hid_dim', 'treshold', 'drop', 'shuflle', 'fscore_home', 'acc', 'fold','model'])
    ind = len(cross_df)
    min_loss = 1
    counter = 0
    while counter<1:
        for batch in bath_sizes:
            data = cross_vali(data,batch)
            for shuflle in shuflle_batch:
                for drop in dropouts:
                    for hid_dim in hidden_dims:
                            #for pos_weight in pos_weights:
                        start_time = time.time()
                        results_df = pd.DataFrame(columns=['epoch', 'batch', 'hid_dim', 'drop', 'shuflle','fscore_home', 'acc', 'fold','model'])
                        new_index = 0
                        set_seed(111)
                        #training_all(data,10,128,0.4,shuflle,100)
                        for fold in range(1,6):
                            results_df, new_index = training(fold,data, batch, hid_dim, tresholds, drop, shuflle, results_df, new_index)
                        check_max_cross_val(results_df,tresholds,batch,hid_dim,data,type,drop)
        counter+=1

