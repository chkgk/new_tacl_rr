import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import joblib
import random
import numpy as np
import sys
import pandas as pd
import copy
from transformers import AdamW
from sklearn import metrics
from mcts_.create_bert_embadding import load_data_and_make_label_sentence, check_max_len,make_tokenization
import time
from mcts_.data_preperation_functions import cross_vali
from transformers import BertTokenizer,BertModel
###############
#batch_size = 5
epoch_num = 100
#hidden_dim = 100
#dropout = 0.5
#treshold = 0.5
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.cuda(1)
loss_type = 'blogic'
#learning_rate = 0.01
###############

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
    #else:
    torch.cuda.manual_seed(seed)

class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()




class BertModel_():

    def __init__(self):
        super(BertModel_, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased", return_dict=True)#, output_attentions = False,encoder_hidden_states=True)
        for n,p in self.bert_model.named_parameters():
            #(n)
            if 'encoder' in n :#and ('10' in n or '11' in n or '9' in n or '8' in n):
                p.requires_grad = True



class BertTagger(nn.Module):

    def __init__(self, hidden_dim, tagset_size,dropout):
        super(BertTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(788, hidden_dim, dropout=dropout)#762
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.bert = BertModel_().bert_model
        self.drop = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        #self.FC = nn.Linear(768, 96)


    def forward(self, sentence_emb,input_ids,attention_masks):
        whole_emb = torch.zeros((int(input_ids.shape[0]/10),10,768)).to(device)#/10
        for batch in range(0,int(input_ids.shape[0]/10)):#
            for i in range(0,10):
                bert_emb = self.bert(input_ids[i+batch*10,:].reshape(1,146),token_type_ids=None,attention_mask=attention_masks[i+batch*10,:].reshape(1,146))#,encoder_hidden_states=True)#[-1][-1] #torch.mean(, dim=1).reshape(Batch,10,768)
                whole_emb[batch,i,:] = bert_emb['pooler_output'].reshape(768)
        whole_emb = self.drop(whole_emb)
        sentence_emb = torch.cat([self.tanh(sentence_emb), whole_emb], dim=2)
        lstm_out, _ = self.lstm(sentence_emb)
        lstm_out = self.drop(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space

class NOLSTMBertTagger(nn.Module):

    def __init__(self, hidden_dim, tagset_size,dropout):
        super(NOLSTMBertTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(788, hidden_dim, dropout=dropout)#762
        self.FC = nn.Linear(788, tagset_size)
        self.bert = BertModel_().bert_model
        self.drop = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        #self.FC = nn.Linear(768, 96)


    def forward(self, sentence_emb,input_ids,attention_masks):
        whole_emb = torch.zeros((int(input_ids.shape[0]/10),10,768)).to(device)#/10
        for batch in range(0,int(input_ids.shape[0]/10)):#
            for i in range(0,10):
                bert_emb = self.bert(input_ids[i+batch*10,:].reshape(1,146),token_type_ids=None,attention_mask=attention_masks[i+batch*10,:].reshape(1,146))#,encoder_hidden_states=True)#[-1][-1] #torch.mean(, dim=1).reshape(Batch,10,768)
                whole_emb[batch,i,:] = bert_emb['pooler_output'].reshape(768)
        whole_emb = self.drop(whole_emb)
        sentence_emb = torch.cat([self.tanh(sentence_emb), whole_emb], dim=2)
        tag_space = self.FC(sentence_emb)
        #lstm_out = self.drop(lstm_out)
        #tag_space = self.hidden2tag(lstm_out)
        return tag_space


class BertTagger_mcts(nn.Module):

    def __init__(self, hidden_dim, tagset_size,dropout):
        super(BertTagger_mcts, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(788, hidden_dim, dropout=dropout)#762
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.bert = BertModel_().bert_model
        self.drop = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        #self.FC = nn.Linear(768, 96)


    def forward(self, sentence_emb,input_ids,attention_masks):
        #whole_emb = torch.zeros((1,input_ids.shape[0],768)).to(device)#/10
        #for batch in range(0,1):#
            #for i in range(0,input_ids.shape[0]):
        bert_emb = self.bert(input_ids,token_type_ids=None,attention_mask=attention_masks)#,encoder_hidden_states=True)#[-1][-1] #torch.mean(, dim=1).reshape(Batch,10,768)

        whole_emb = self.drop(bert_emb['pooler_output'])
        whole_emb=whole_emb.reshape([1, whole_emb.shape[0], whole_emb.shape[1]])
        sentence_emb = torch.cat([self.tanh(sentence_emb), whole_emb], dim=2)
        lstm_out, _ = self.lstm(sentence_emb)
        lstm_out = self.drop(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space

class BertOnlyTagger(nn.Module):

    def __init__(self, hidden_dim, tagset_size,dropout):
        super(BertOnlyTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(768, hidden_dim, dropout=dropout)#762
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.bert = BertModel_().bert_model
        self.drop = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        #self.FC = nn.Linear(768, 96)


    def forward(self, sentence_emb,input_ids,attention_masks):
        whole_emb = torch.zeros((int(input_ids.shape[0]/10),10,768)).to(device)#/10
        for batch in range(0,int(input_ids.shape[0]/10)):#
            for i in range(0,10):
                bert_emb = self.bert(input_ids[i+batch*10,:].reshape(1,146),token_type_ids=None,attention_mask=attention_masks[i+batch*10,:].reshape(1,146))#,encoder_hidden_states=True)#[-1][-1] #torch.mean(, dim=1).reshape(Batch,10,768)
                whole_emb[batch,i,:] = bert_emb['pooler_output'].reshape(768)
        whole_emb = self.drop(whole_emb)
        #sentence_emb = torch.cat([self.tanh(sentence_emb), whole_emb], dim=2)
        lstm_out, _ = self.lstm(whole_emb)
        lstm_out = self.drop(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space



def crete_random_batches(train_df, batch_size,training_data,shuflle_batch):
    if shuflle_batch==False:
        return training_data
    training_data = []
    count = 0
    batch=[]
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    for index, row in train_df.iterrows():
        game_features = [row[col] for col in list(train_df.columns) if 'features_round' in col]
        text_features = [row[col] for col in list(train_df.columns) if 'review_round' in col]
        labels = row['labels']#[np.int_(1) if val == 'hotel' else np.int_(0) for val in row['labels']]
        batch.append((game_features,text_features, labels))
        count += 1
        if count == batch_size or index==len(train_df)-1:
            training_data.append(batch)
            batch = []
            count = 0
    return training_data

def crete_random_batches_for_test(test):
    training_data = []
    count = 0
    batch=[]
    for index, row in test.iterrows():
        game_features = [row[col] for col in list(test.columns) if 'features_round' in col]
        text_features = [row[col] for col in list(test.columns) if 'review_round' in col]
        try:
            labels = row['labels']#[np.int_(1) if val == 'hotel' else np.int_(0) for val in row['labels']]
        except:
            labels = []
        training_data.append((game_features,text_features, labels))
    return training_data



def training(fold,data,batch_size,hidden_dim,treshold,dropout,shuflle_batch,results_df, new_index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = nn.BCEWithLogitsLoss()#pos_weight=torch.tensor([pos_weight]).to(device))  # pos_weight=torch.tensor([0.9]))#MSELoss()#
    model = BertTagger(hidden_dim=hidden_dim, tagset_size=1,dropout=dropout)
    model.to(device)
    train_df = data[data['fold']!=fold]
    test = data[data['fold']==fold]
    epoch_counter, min_loss=0,100
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    testing_data = crete_random_batches_for_test(test)
    for n, p in model.named_parameters():
        if 'bert' in n:
            p.requires_grad = False
    optimizer_lstm = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)#, momentum=0.9)#optim.Adam(model.parameters(), lr=0.0001)
    optimizer = optimizer_lstm
    for epoch in range(epoch_num):
        if epoch==8:
            for n, p in model.named_parameters():
                if 'bert' in n:
                    p.requires_grad = True
            optimizer = AdamW(model.bert.parameters(), lr=1e-5)
        tot_loss =0
        epoch_counter+=1
        training_data = crete_random_batches(train_df, batch_size,[],shuflle_batch)
        i=0
        for batch in training_data:
            sentence_game = [sample[0] for sample in batch]
            sentence_text = [sample[1] for sample in batch]
            tags_ = [sample[2] for sample in batch]
            input_ids, attention_masks = make_tokenization(sentence_text, 146, tokenizer,False)
            real_teg = torch.Tensor(tags_).to(device)
            sentence_game = torch.Tensor(sentence_game).to(device)
            attention_masks = attention_masks.to(device)
            input_ids = input_ids.to(device)
            model.zero_grad()
            tag_scores = model(sentence_game,input_ids,attention_masks)
            loss = loss_function(tag_scores.reshape(tag_scores.shape[0],10), real_teg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            tot_loss+=loss.item()
            i+=1
        print('training loss:',tot_loss/len(training_data),type(optimizer))
        if epoch_counter>=5:
            # torch.save(model.state_dict(),
            #       f'bert_models2/4final_model_{fold}_{batch_size}_hid_dim_{hidden_dim}_drop_{dropout}_epoch_{epoch_counter}.th')
            #
            # print('save!!!!')
            model.eval()
            with torch.no_grad():
                sentence_game = [sample[0] for sample in testing_data]
                sentence_text = [sample[1] for sample in testing_data]
                tags_ = [sample[2] for sample in testing_data]
                input_ids, attention_masks = make_tokenization(sentence_text, 146, tokenizer)
                real_teg = torch.Tensor(tags_).to(device)
                sentence_game = torch.Tensor(sentence_game).to(device)
                attention_masks = attention_masks.to(device)
                input_ids = input_ids.to(device)
                tag_scores = model(sentence_game,input_ids,attention_masks)
                loss_test = loss_function(tag_scores.reshape(real_teg.shape[0], 10), real_teg).item()
                print('test_loss: ',loss_test ,type(optimizer))
                tag_scores=torch.sigmoid(tag_scores)
                treshold_ = 0.5
                y, pred = list(np.array(real_teg.reshape(real_teg.shape[0] * real_teg.shape[1]).cpu())),[1 if val>=treshold_ else 0 for val in list(np.array(tag_scores.reshape(real_teg.shape[0] * real_teg.shape[1]).cpu()))]
                f_score_home = metrics.f1_score(y, pred,pos_label=0)
                acc = metrics.accuracy_score(y, pred)
                results_df.loc[new_index]=[epoch_counter,batch_size,hidden_dim,treshold_,dropout,shuflle_batch,f_score_home,acc,fold, 0,loss_test]#copy.deepcopy(model).state_dict()]
                new_index+=1
            model.train()
    return results_df,new_index



def training_all(data,batch_size,hidden_dim,dropout,shuflle_batch):
    set_seed(76)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = nn.BCEWithLogitsLoss()#pos_weight=torch.tensor([pos_weight]).to(device))  # pos_weight=torch.tensor([0.9]))#MSELoss()#
    model = LSTMBertTagger(hidden_dim=hidden_dim, tagset_size=1,dropout=dropout)
    model.to(device)
    train_df = data#[data['fold']!=fold]
    #test = data[data['fold']==fold]
    epoch_counter, min_loss=0,100
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    #testing_data = crete_random_batches_for_test(test)
    for n, p in model.named_parameters():
        if 'bert' in n:
            p.requires_grad = False
    optimizer_lstm = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)#, momentum=0.9)#optim.Adam(model.parameters(), lr=0.0001)
    optimizer = optimizer_lstm
    for epoch in range(epoch_num):
        if epoch==8:
            for n, p in model.named_parameters():
                if 'bert' in n:
                    p.requires_grad = True
            optimizer = AdamW(model.bert.parameters(), lr=1e-5)
        tot_loss =0
        epoch_counter+=1
        training_data = crete_random_batches(train_df, batch_size,[],shuflle_batch)
        i=0
        for batch in training_data:
            sentence_game = [sample[0] for sample in batch]
            sentence_text = [sample[1] for sample in batch]
            tags_ = [sample[2] for sample in batch]
            input_ids, attention_masks = make_tokenization(sentence_text, 146, tokenizer,False)
            real_teg = torch.Tensor(tags_).to(device)
            sentence_game = torch.Tensor(sentence_game).to(device)
            attention_masks = attention_masks.to(device)
            input_ids = input_ids.to(device)
            model.zero_grad()
            tag_scores = model(sentence_game,input_ids,attention_masks)
            loss = loss_function(tag_scores.reshape(tag_scores.shape[0],10), real_teg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            tot_loss+=loss.item()
            i+=1
        print('training loss:',tot_loss/len(training_data),type(optimizer))
        if epoch_counter==12:#>=5:
            torch.save(model.state_dict(),
                   f'NOLSTM_TACL/DM-BERT-MODEL_{batch_size}_hid_dim_{hidden_dim}_drop_{dropout}_epoch_{epoch_counter}.th')

            print('save!!!!')

    return results_df,new_index

def check_max_cross_val(df_all_thres,tresholds,batch,hid_dim,drop):
        df = df_all_thres
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
        df_['loss_avg'] = df_.apply(lambda x: (x['loss_1'] + x['loss_2'] + x['loss_3'] + x[
            'loss_4'] + x['loss_5']) / 5, axis=1)
        min_until_now_loss=min(df_['loss_avg'].values)
        max_until_now_fscore = max(df_['avg_f_score_home'].values)
        #max_until_now_fscore = max(df_['avg_f_score_home'].values)
        max_until_now_fscore = copy.deepcopy(df_[df_['loss_avg']==min_until_now_loss]['avg_f_score_home'].values[0])
        #min_until_now_loss  = copy.deepcopy(df_[df_['avg_f_score_home']==max_until_now_fscore]['loss_avg'].values[0])
        epoch = copy.deepcopy(df_[df_['loss_avg']==min_until_now_loss]['epoch'].values[0])
        print(min_until_now_loss)
        #with open("bert_models2/bert_results_text_and_features_31.12_by_loss.txt", "a") as myfile:
        with open("bert_models2/bert_results_20_features_10.1_by_loss.txt", "a") as myfile:

            myfile.write(f'loss: {min_until_now_loss},f1_home: {str(max_until_now_fscore)},epoch: {epoch},batch: {batch},hiddim: {hid_dim}, drop: {drop}')
            myfile.write('\n')
        print('save!!!!!!!!!!!!!!!!!!!!')



def crf_eval(test,model,model_type='dm'):
    batch = []
    with torch.no_grad():
        for index,row in test.iterrows():
                features = [row[col] for col in list(test.columns) if 'features_round' in col]
                labels = row['labels']#[np.int_(1) if val == 'hotel' else np.int_(0) for val in row['labels']]
                batch.append((features, labels))
        sentence = [sample[0] for sample in batch]
        #sentence = sentence.reshape([1, sentence.shape[0], sentence.shape[1]])
        #model.zero_grad()
        model.to(device)
        sentence = torch.Tensor(sentence).to(device)
        if model_type=='dm':
            tag_scores = torch.sigmoid(model(sentence))#.reshape(1,10).cpu()[0].tolist()  # .view(len(sentence), 1, -1))
        else:
            tag_scores = model(sentence)#.reshape(1,10).cpu()[0].tolist()  # .view(len(sentence), 1, -1))
        return tag_scores

def bert_eval_mcts(test,model,model_type='dm'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    testing_data = crete_random_batches_for_test(test)
    batch = testing_data
    sentence_game = [sample[0] for sample in batch]
    sentence_text = [sample[1] for sample in batch]
                    # Step 1. Remember that Pytorch accumulates gradients.
                    # We need to clear them out before each instance
    input_ids, attention_masks = make_tokenization(sentence_text, 146, tokenizer)
    sentence_game = torch.Tensor(sentence_game).to(device)
    attention_masks = attention_masks.to(device)
    input_ids = input_ids.to(device)
    model.eval()
    model.cuda()
    if model_type=='dm':
        tag_scores = torch.sigmoid(model(sentence_game,input_ids,attention_masks))
    else:
        tag_scores = model(sentence_game,input_ids,attention_masks)
    #tag_scores = [float(i) for val in tag_scores.cpu().detach().numpy() for i in val]
    #tag_scores2 = torch.sigmoid(model(sentence_game[50:],input_ids[50:],attention_masks[50:]))

    return tag_scores[0][-1].item()


def bert_eval(test,model,model_type='dm'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    testing_data = crete_random_batches_for_test(test)
    batch = testing_data
    sentence_game = [sample[0] for sample in batch]
    sentence_text = [sample[1] for sample in batch]
                    # Step 1. Remember that Pytorch accumulates gradients.
                    # We need to clear them out before each instance
    input_ids, attention_masks = make_tokenization(sentence_text, 146, tokenizer)
    sentence_game = torch.Tensor(sentence_game).to(device)
    attention_masks = attention_masks.to(device)
    input_ids = input_ids.to(device)
    model.eval()
    model.cuda()
    tot_tag_scores=[]
    for i in range(0,101):
        if model_type=='dm':
            tag_scores = torch.sigmoid(model(sentence_game[i:i+1,:,:],input_ids[i*10:i*10+10],attention_masks[i*10:i*10+10]))
        else:
            tag_scores = model(sentence_game[i:i+1,:,:],input_ids[i*10:i*10+10],attention_masks[i*10:i*10+10])
        tag_scores = [float(i) for val in tag_scores.cpu().detach().numpy() for i in val]
        tot_tag_scores = tot_tag_scores + tag_scores
    #tag_scores2 = torch.sigmoid(model(sentence_game[50:],input_ids[50:],attention_masks[50:]))

    return tot_tag_scores

if __name__ == '__main__':
    bath_sizes = [5]
    hidden_dims = [64,128,256]
    tresholds = [0.5]
    dropouts = [0.3,0.4,0.5,0.6]
    shuflle_batch = [True]
    cross_df = pd.DataFrame(columns=['param', 'f_home_max', 'acc', 'epoch'])
    new_index = 0
    data = load_data_and_make_label_sentence('results_payments_status_train.csv')
    game_features = joblib.load('embaddings/10.1train_for_bert_with_expert_behavioral_features.pkl')  # embaddings/new_maya_emb_for_val_new_features_{type}.pkl')#_with_bert#('1-10_bert_and_manual_avg_history_embadding.pkl')
    game_features = game_features.drop(['labels'], axis=1)
    game_features['labels'] = game_features['labels_for_probability']
    del game_features['labels_for_probability']

    game_features = cross_vali(game_features, 10)
    print(len(game_features), len(data))
    all_data = game_features.merge(data, on=['pair_id'], how='left')
    print(len(all_data))
    del all_data['labels_y']
    del all_data['labels_for_probability']
    all_data['labels'] = all_data['labels_x']
    del all_data['labels_x']
    ind = len(cross_df)
    counter = 0
    f_home_max, acc_max = -10000, -10000
    for batch in bath_sizes:
            data = cross_vali(data,10)
            for shuflle in shuflle_batch:
                for drop in dropouts:
                    for hid_dim in hidden_dims:
                        start_time = time.time()
                        results_df = pd.DataFrame(columns=[ 'epoch', 'batch', 'hid_dim', 'treshold', 'drop', 'shuflle','fscore_home', 'acc', 'fold','model','loss'])
                        new_index = 0
                        training_all(all_data, batch, 64, 0.6, shuflle)
                        #for fold in range(1,5):#(1,6):
                        #    results_df, new_index = training(fold,all_data, batch, hid_dim, tresholds, drop, shuflle, results_df, new_index)
                        #check_max_cross_val(results_df,tresholds,batch,hid_dim,drop)


