import random
import pandas as pd
def make_data_to_seq_reg(data):
    columns = list(data.columns)
    # columns.remove('raisha')
    columns.remove('sample_id')
    columns.remove('features_round_10')
    new_df = pd.DataFrame(columns=columns)
    index_len = 0
    for index, row in data.iterrows():
        labels = [np.int_(1) if val == 'hotel' else np.int_(0) for val in row['labels']]
        for i in range(10, 11):
            for j in range(1, i):
                new_df.at[index_len, f'features_round_{j}'] = row[f'features_round_{j}'][:-2]+[j]+[sum(labels[:j])]#+labels[:j]+[0]*(9-len(labels[:j]))#
            new_df.at[index_len, 'labels'] = [sum(labels[y:]) for y in range(1,10)]  # ,labels[i-2])#[sum(labels[y-1:]) for y in range(1,11)]
            new_df.at[index_len, 'pair_id'] = row['pair_id']#[sum(labels[y-1:]) for y in range(1,11)]
            new_df.at[index_len, 'raisha'] = i - 2
            index_len += 1
    return new_df

def make_data_to_reg(data):#,only_round=None
    new_df = pd.DataFrame(columns=['x','labels','pair_id'])
    index_len = 0
    #if only_round==None:
    for index,row in data.iterrows():
        labels = row['labels']#[np.int_(1) if val == 'hotel' else np.int_(0) for val in row['labels']]
            #for i in range(2,11):
        for j in range(1,10):#+[j]#row[f'features_round_{j}']+[-20:]#row[f'features_round_{j}'][-20:]+
                        #new_df.at[index_len,f'features_round_{j}'] = row[f'features_round_{j}']#[-20:]+[sum(labels[:j])] + labels[:j]  + [0]*(9-len(labels[:j]))#[-20:]
            new_df.at[index_len,'x'] = row[f'features_round_{j}']#[-20:]+[sum(labels[:j])] + labels[:j]  + [0]*(9-len(labels[:j]))#[-20:]
            try:
                new_df.at[index_len, 'labels'] = labels[j-1]#,labels[i-2])#[sum(labels[y-1:]) for y in range(1,11)]
            except:
                new_df.at[index_len, 'labels'] = labels
            new_df.at[index_len,'pair_id'] = row['pair_id']

            index_len+=1
    return new_df

def make_data_to_reg_for_simulation(data):#,only_round=None
    new_df = pd.DataFrame(columns=['x','labels','pair_id'])
    index_len = 0
    #if only_round==None:
    for index,row in data.iterrows():
        labels = row['labels']#[np.int_(1) if val == 'hotel' else np.int_(0) for val in row['labels']]
            #for i in range(2,11):
        for j in range(1,10):#+[j]#row[f'features_round_{j}']+[-20:]#row[f'features_round_{j}'][-20:]+
                        #new_df.at[index_len,f'features_round_{j}'] = row[f'features_round_{j}']#[-20:]+[sum(labels[:j])] + labels[:j]  + [0]*(9-len(labels[:j]))#[-20:]
            new_df.at[index_len,'x'] = row[f'features_round_{j}']#[-20:]+[sum(labels[:j])] + labels[:j]  + [0]*(9-len(labels[:j]))#[-20:]
            try:
                new_df.at[index_len, 'labels'] = labels[j-1]#,labels[i-2])#[sum(labels[y-1:]) for y in range(1,11)]
            except:
                new_df.at[index_len, 'labels'] = labels
            new_df.at[index_len,'pair_id'] = row['pair_id']
            index_len+=1
            if index_len == len(data.columns)-2:
                return new_df


def cross_vali(df,batch):
    random.seed(batch)
    ids = list(df['pair_id'].unique())
    random.shuffle(ids)
    ids1, ids2, ids3, ids4, ids5 = ids[:80], ids[80:160], ids[160:240], ids[240:320], ids[320:]
    df['fold'] = 0
    for index, row in df.iterrows():
        if row['pair_id'] in ids1:
            df.at[index, 'fold'] = 1
        elif row['pair_id'] in ids2:
            df.at[index, 'fold'] = 2
        elif row['pair_id'] in ids3:
            df.at[index, 'fold'] = 3
        elif row['pair_id'] in ids4:
            df.at[index, 'fold'] = 4
        else:
            df.at[index, 'fold'] = 5
    return df

def crete_random_batches(train_df, batch_size,training_data,shuflle_batch,embedding_dim):
    if shuflle_batch==True:
        train_df = train_df.sample(frac=1)
    training_data = []
    count = 0
    batch=[]
    train_df = train_df.reset_index(drop=True)
    columns = [col for col in list(train_df.columns) if ('x' in col or 'features_round' in col)]#'features_round' in col]
    for index, row in train_df.iterrows():
        features = [row[col] for col in columns  if str(row[col]) != 'nan' ]#else [0] * embedding_dim for col in columns
        labels = row['labels']
        batch.append((features, labels))
        count += 1
        if count == batch_size or index==len(train_df)-1:
            training_data.append(batch)
            batch = []
            count = 0
    return training_data
