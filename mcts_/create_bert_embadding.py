import pandas as pd
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split
import torch
import random
import joblib
import numpy as np
import time
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 10
epochs=100


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output#[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def load_reviews_data_mcts(data,current_round):
    features = pd.DataFrame(columns=['total_cum_features'])
    features['total_cum_features'] = data.apply(
                lambda row: row['group_sender_answer_reviews'], axis=1)
    new_column_name = {}
    for val in range(0, current_round):
        new_column_name[val] = f"review_round_{val + 1}"
    features.reset_index(drop=True, inplace=True)
    features = features.T.rename(columns=new_column_name)
    features = features.reset_index(drop=True)
    features.at[0, 'pair_id'] = data.at[0,'pair_id']
    return features




def load_data_and_make_label_sentence(file_name):
    data = pd.read_csv(file_name)
    data = data.loc[(data.status == 'play') & (data.player_id_in_group == 2)]
    # print(f'Number of rows in data: {self.data.shape[0]} after keep only play and decision makers')
    data = data.drop_duplicates()
    data['exp_payoff'] = data.group_receiver_choice.map({1: 0, 0: 1})
    total_exp_payoff = data.groupby(by='pair_id').agg(
        total_exp_payoff=pd.NamedAgg(column='exp_payoff', aggfunc=sum))
    data = data.merge(total_exp_payoff, how='left', right_index=True, left_on='pair_id')
    final_data = pd.DataFrame()
    for pair in data['pair_id'].unique():
        data_pair = data[data['pair_id'] == pair]
        data_pair['next_exp_payoff'] = data_pair['total_exp_payoff'] - (data_pair['exp_payoff'].cumsum() - data_pair['exp_payoff'])
        features = pd.DataFrame(columns=['total_cum_features'])
        features['total_cum_features'] = data_pair.apply(
                lambda row: row['group_sender_answer_reviews'], axis=1)
        new_column_name = {}
        for val in range(0, 10):
            new_column_name[val] = f"review_round_{val + 1}"
        features.reset_index(drop=True, inplace=True)
        features = features.T.rename(columns=new_column_name)
        features = features.reset_index(drop=True)
        features.at[0, 'pair_id'] = pair
        # features['label_total_exp_payoff'] = None
        # features['label_total_exp_payoff'] = features['label_total_exp_payoff'].astype(object)
        features['labels'] = None
        features['labels'] = features['labels'].astype(object)
        # features.at[0,'label_total_exp_payoff'] = list(data_pair['total_exp_payoff'].values)
        features.at[0, 'labels'] = list(data_pair['next_exp_payoff'].values)
        features['labels_for_probability'] = None
        features['labels_for_probability'] = features['labels_for_probability'].astype(object)
        features.at[0, 'labels_for_probability'] = list(data_pair['exp_payoff'].values)
        final_data = pd.concat([final_data, features], axis=0, ignore_index=True)
    return final_data
def check_max_len(data,tokenizer):
    max_len=0
    for i in range(1,11):
        for sent in list(data[f'review_round_{i}'].values):
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Update the maximum sentence length.
            max_len = max(max_len, len(input_ids))
    return max_len
def make_tokenization(sentences,max_len,tokenizer,label=False):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
    # For every sentence...
    for sentences_ in sentences:
        for sent in sentences_:
    # for index,row in sentences.iterrows():
    #      for i in range(1,11):
            #sent = [f'review_round_{i}']
            encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',
                truncation=True# Return pytorch tensors.
            )

        # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    if label==True:
        labels = torch.tensor([l for val in list(sentences['labels'].values) for l in val])
        return input_ids, attention_masks, labels
    else:
        return input_ids, attention_masks
    # Print sentence 0, now as a list of IDs.
    #print('Original: ', sent)
    #print('Token IDs:', input_ids)

def make_tokenization2(sentences,max_len,tokenizer,label=False):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
    # For every sentence...
    #for sentences_ in sentences:
    #    for sent in sentences_:
    for index,row in sentences.iterrows():
         #for i in range(1,11):
            sent = row['review']
            encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',
                truncation=True# Return pytorch tensors.
            )

        # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    if label==True:
        labels = torch.tensor([l for val in list(sentences['labels'].values) for l in val])
        return input_ids, attention_masks, labels
    else:
        return input_ids, attention_masks
    # Print sentence 0, now as a list of IDs.
    #print('Original: ', sent)
    #print('Token IDs:', input_ids)

def split_create_train_bal(input_ids, attention_masks ,labels):
    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_size = int(0.99 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )
    return train_dataloader, validation_dataloader

def training(model,optimizer,scheduler,train_dataloader,tokenizer):
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.   '.format(step, len(train_dataloader)))
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()
            outputs = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)
            loss= outputs.loss
            #logits = outputs.logits



            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.

        print("")
        print(f"Average training loss: {avg_train_loss}")
        if avg_train_loss<=0.45:
            model.save_pretrained("BertForSequenceClassificationmodel")
            tokenizer.save_pretrained("BertForSequenceClassificationtokenizer")
            print('save!!')
            return model,tokenizer

def bert_training():
    data = load_data_and_make_label_sentence()
    del data['labels']
    data['labels'] = data['labels_for_probability']
    del data['labels_for_probability']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    max_len = check_max_len(data, tokenizer)
    input_ids, attention_masks, labels = make_tokenization(data, max_len, tokenizer,True)
    train_dataloader, validation_dataloader = split_create_train_bal(input_ids, attention_masks, labels)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=True,
        return_dict=True)
    model.cuda()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs)
    model,tokenizer = training(model, optimizer, scheduler, train_dataloader,tokenizer)
    return model,tokenizer
def extract_hidden(model,tokenizer):
    #model = BertForSequenceClassification.from_pretrained('BertForSequenceClassificationmodel')
    #tokenizer = BertTokenizer.from_pretrained('BertForSequenceClassificationtokenizer')
    #model.cuda()
    data = pd.read_csv('hotels_index_test_data.csv')
    max_len = 146#check_max_len(data, tokenizer)
    input_ids, attention_masks = make_tokenization2(data, max_len, tokenizer,label=False)
    model.eval()
    data['embadding'] = None
    ind = 0
    with torch.no_grad():
        for i in range(0,len(input_ids)):
            b_input_ids = input_ids[i].reshape(1,146).to(device)
            b_input_mask = attention_masks[i].reshape(1,146).to(device)
            #b_labels = batch[2].to(device)
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)
            #print(outputs.keys())
            data.at[ind,'embadding'] =  mean_pooling(outputs[-1][-1],b_input_mask)
            ind+=1
    joblib.dump(data, 'new_bert_embadding_for_clas.pkl')

if __name__ == '__main__':
    model,tokenizer = bert_training()
    extract_hidden(model,tokenizer)