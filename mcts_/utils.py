import pandas as pd
import ast
#import matplotlib.pyplot  as plt
def f1():
    df_indexes = pd.read_csv('hotels_index_test_data.csv')
    df_game = pd.read_csv('results_payments_status_test.csv')
    df_game = df_game[(df_game.pair_id == 'tlyunc45_6') & (df_game.player_id_in_group == 1)][
        [col for col in df_game.columns if
         'group_random_positive_negative_review' in col or 'group_average_score' in col]].reset_index(drop=True)
    columns_ = [val for val in df_game.columns if 'pos' in val]
    for col in columns_:
        df_game[col] = df_game.apply(lambda x: x[col].replace('.', '').replace(' ', '').replace('\n', ''), axis=1)
    df_indexes['review'] = df_indexes.apply(lambda x: x['review'].replace('.', '').replace(' ', '').replace('\n', ''),
                                            axis=1)
    for i in range(0, 7):
        df_indexes.columns = ['review_id', f'group_random_positive_negative_review_{i}']
        df_game = df_game.merge(df_indexes,
                                on=f'group_random_positive_negative_review_{i}',
                                how='left')
        df_game[f'review_id_{i}'] = df_game[f'review_id']
        del df_game['review_id']
    df_game.to_csv('reviews_and_index.csv')


def f2():
    data = pd.read_csv('results_payments_status_test.csv')
    data = data.loc[(data.status == 'play') & (data.player_id_in_group == 2)]
    # print(f'Number of rows in data: {self.data.shape[0]} after keep only play and decision makers')
    data = data.drop_duplicates()
    data = data[['pair_id', 'subsession_round_number', 'group_sender_answer_reviews',
                  'group_lottery_result', 'review_id', 'previous_round_lottery_result',
                 'group_sender_answer_index',
                 'previous_round_decision', 'previous_review_id', 'group_average_score',
                 'lottery_result_low', 'lottery_result_med1', 'previous_round_lottery_result_low',
                 'previous_round_lottery_result_high', 'previous_average_score_low',
                 'previous_average_score_high', 'previous_round_lottery_result_med1',
                 'group_sender_payoff', 'lottery_result_high',
                 'chose_lose', 'chose_earn', 'not_chose_lose', 'not_chose_earn',
                 'previous_score', 'group_sender_answer_scores']]
    for hotel in data['group_average_score'].unique():
        dic={}
        for ind in data['group_sender_answer_index'].unique():
            data_ = data[(data.group_average_score==hotel) & (data.group_sender_answer_index==ind) ]
            dic[ind] = len(data_)
        print(hotel, dic)

"""
group_average_score-8.5+,group_average_score 7.5-8.5, group_average_score 7.5-0, group_sender_answer_scores>9.5, group_sender_answer_scores9.5-7.5includ, group_sender_answer_scores<7.5
group_sender_answer_index<=3,group_sender_answer_index>3
"""

def f3(df_dm,df_q,real_user):
    all_interaction = pd.read_csv('results_payments_status_test.csv')
    all_interaction = all_interaction.loc[(all_interaction.status == 'play') & (all_interaction.player_id_in_group == 2)]
    all_interaction = all_interaction.drop_duplicates()
    var1, var2 = {},{}
    for val in all_interaction['group_average_score'].unique():
        dic_dm = {0:0,1:0,2:0,3:0,4:0,5:0,6:0}
        dic_q = {0:0,1:0,2:0,3:0,4:0,5:0,6:0}
        dic_real = {0:0,1:0,2:0,3:0,4:0,5:0,6:0}

        for i in range(0,548):
            pair = df_dm.at[i,'pair_id']
            df_game_pair = all_interaction[(all_interaction.pair_id==pair) & (all_interaction.group_average_score==val)]
            round = df_game_pair['subsession_round_number'].values[0]
            #print(round)
            score_dm = ast.literal_eval(df_dm.at[i,'actions'])[round-1]
            score_q = ast.literal_eval(df_q.at[i,'actions'])[round-1]
            score_real = ast.literal_eval(real_user.at[i,'actions'])[round-1]
            if score_dm in dic_dm:
                dic_dm[score_dm]+=1
            else:
                dic_dm[score_dm]=1
            if score_q in dic_q:
                dic_q[score_q]+=1
            else:
                dic_q[score_q]=1
            if score_real in dic_real:
                dic_real[score_real]+=1
            else:
                dic_real[score_real]=1
        #print(dic_dm)
        import statistics
        new_df = pd.DataFrame(columns=['rev','dm_count','q_count','real_user'])
        new_df['rev'] = list(dic_dm.keys())
        new_df['dm_softmax'] = list(dic_dm.values())
        new_df['mcts_agent'] = list(dic_q.values())
        new_df['real_user'] = list(dic_real.values())
        for_var_dic_real = [0]*list(dic_real.values())[0]+[1]*list(dic_real.values())[1]+[2]*list(dic_real.values())[2]+[3]*list(dic_real.values())[3]+[4]*list(dic_real.values())[4]+[5]*list(dic_real.values())[5]+[6]*list(dic_real.values())[6]
        for_var_dic_model = [0]*list(dic_q.values())[0]+[1]*list(dic_q.values())[1]+[2]*list(dic_q.values())[2]+[3]*list(dic_q.values())[3]+[4]*list(dic_q.values())[4]+[5]*list(dic_q.values())[5]+[6]*list(dic_q.values())[6]
        print(val,'real_user_actions_var:',statistics.variance(for_var_dic_real),'model_actions_var:',statistics.variance(for_var_dic_model))
        new_df = new_df.sort_values('rev')
        var1[val] = statistics.variance(for_var_dic_real)
        var2[val] = statistics.variance(for_var_dic_model)
        #new_df.plot('rev',['mcts_agent','real_user'],width=0.2,kind='bar')
        #plt.bar(dic_q.keys(),dic_q.values(),width=0.2)111111111
        #plt.title(val)
        #plt.savefig(f'mcts_results/1_{val}_mcts_real_baisline.png')
        #plt.close()
    new_df =pd.DataFrame(columns=['hotel','real_user_var','model_var'])
    new_df['hotel']=var1.keys()
    new_df['real_user_var']=var1.values()
    new_df['model_var']=var2.values()
    new_df = new_df.sort_values('hotel')
    new_df.plot('hotel',['real_user_var','model_var'],width=0.2,kind='bar')
    plt.title(val)
    plt.savefig(f'mcts_results/var_{val}_mcts_real_baisline.png')
    plt.close()

def f4(hotel):
    all_interaction = pd.read_csv('results_payments_status_test.csv')
    all_interaction = all_interaction.loc[
        (all_interaction.status == 'play') & (all_interaction.player_id_in_group == 2)]
    all_interaction = all_interaction.drop_duplicates()

    #for val in all_interaction['group_average_score'].unique():
    dic = {}
    all_interaction_hotel = list(all_interaction[all_interaction['group_average_score']==hotel]['group_sender_answer_index'].values)
        #print(all_interaction_hotel)
    for i in range(1,8):
        count = all_interaction_hotel.count(i)
        if count>5:
            dic[i-1] = all_interaction_hotel.count(i)
    return list({k: v for k, v in sorted(dic.items(), key=lambda item: item[1])}.keys())
        #plt.bar(dic.keys(),dic.values(),label = val)
        #plt.show()
def f5(mcts_norollout,real_user):
    all_interaction = pd.read_csv('results_payments_status_test.csv')
    all_interaction = all_interaction.loc[
        (all_interaction.status == 'play') & (all_interaction.player_id_in_group == 2)]
    all_interaction = all_interaction.drop_duplicates()
    mcts_lie = {}
    score_real_lie = {}
    for val in all_interaction['group_average_score'].unique():
        mcts_lie[val]=0
        score_real_lie[val]=0
        for i in range(0, 548):
            pair = mcts_norollout.at[i, 'pair_id']
            df_game_pair = all_interaction[
                (all_interaction.pair_id == pair) & (all_interaction.group_average_score == val)]
            round = df_game_pair['subsession_round_number'].values[0]
            # print(round)
            score_mcts_norollout = df_game_pair[f'group_score_{ast.literal_eval(mcts_norollout05.at[i, "actions"])[round - 1]}'].values[0]
            score_real = df_game_pair[f'group_score_{ast.literal_eval(real_user.at[i, "actions"])[round - 1]}'].values[0]
            max_score =  df_game_pair[f'group_score_6'].values[0]
            mcts_lie[val] += (score_mcts_norollout-val)/(max_score-val)
            score_real_lie[val] += (score_real-val)/(max_score-val)

    for k in mcts_lie.keys():
        mcts_lie[k] = mcts_lie[k]/len(real_user)
        score_real_lie[k] = score_real_lie[k]/len(real_user)
    mcts_lie = sorted(mcts_lie.items(), key=lambda item: item[0])
    score_real_lie = sorted(score_real_lie.items(), key=lambda item: item[0])
    print(mcts_lie)
    print(score_real_lie)
    plt.scatter([val[0] for val in mcts_lie],[val[1] for val in mcts_lie],label = 'mcts_')
    plt.scatter([val[0] for val in mcts_lie],[val[1] for val in score_real_lie],label = 'real_expert')
    plt.legend()
    plt.savefig('l1.png')

def df6(df_user):
    all_interaction = pd.read_csv('results_payments_status_test.csv')
    all_interaction = all_interaction.loc[
        (all_interaction.status == 'play') & (all_interaction.player_id_in_group == 2)]
    all_interaction = all_interaction.drop_duplicates()
    new_df = pd.DataFrame(columns = ['pair_id','real_dm','dm_model'])
    counter=0
    for pair in all_interaction['pair_id'].unique():
        real_user =all_interaction[all_interaction['pair_id']==pair]['group_sender_payoff'].values
        pair_df_user = df_user[df_user['pair_id']==pair]
        dm_tot = [0,0,0,0,0,0,0,0,0,0]
        for index,row in pair_df_user.iterrows():
            dm_actions = ast.literal_eval(row['total_payoff'])
            # for i in range(0,10):
            #     dm_tot[i] =  dm_tot[i]+dm_actions[i]
            break
        new_df.loc[counter] = [pair, real_user,dm_actions]
        counter+=1
    return new_df



def f7():
    import os
 # ,0.53,0.6,0.52,0.56,0.57]#],0.7,0.75]#[0.63,0.65,0.7,0.75,0.73,0.67,0.77,0.8,0.83,0.5,0.55,0.6,0.53]
    dropouts = [0.4, 0.3, 0, 0.5, 0.2, 0.6]
    bath_sizes =[5, 10, 15, 25, 30, 20, 35, 40, 50, 45]# [ 10, 15, 25, 30, 20, 35, 40, 50, 45]
    hidden_dims = [30, 40, 50, 60, 70,100,150,200,50,250,300]#[30, 40, 50, 60, 70, 90]  # ,[50,80,100]
    dropouts = [0.4, 0.3, 0, 0.5, 0.2,0.6]
    directory = r'bert_models2'
    for b in bath_sizes:
        for hd in hidden_dims:
            for d in dropouts:
                loss,count=0,0
                for file in os.listdir(directory):
                    if ('_0.0001_' in file) and (f'{b}_hid_dim_{hd}_drop_{d}' in file) and ('loss' in file) :
                        print(file)
                        loss += float(file.split('loss_')[-1].split('.th')[0])
                        #print(file)
                        #print(float(file.split('loss_')[-1].split('.th')[0]))
                        count+=1
                if count==5 and loss/count<=0.456:#loss/count<=0.45:
                    print(loss/count,d,hd,b)

if __name__ == '__main__':
    """
    df_dm_hard = pd.read_csv('/home/mayatarno/PycharmProjects/models_and_mcts/mcts_results/baisline_16.12_dm_hard.csv')
    df_dm_softmax = pd.read_csv('/home/mayatarno/PycharmProjects/models_and_mcts/mcts_results/baisline_16.12_dm_softmax.csv')
    df_only3 = pd.read_csv('/home/mayatarno/PycharmProjects/models_and_mcts/mcts_results/baisline_16.12_only3.csv')
    df_only6 = pd.read_csv('/home/mayatarno/PycharmProjects/models_and_mcts/mcts_results/baisline_16.12_only6.csv')
    df_q_hard = pd.read_csv('/home/mayatarno/PycharmProjects/models_and_mcts/mcts_results/baisline_16.12_q_hard.csv')
    df_q_softmax = pd.read_csv('/home/mayatarno/PycharmProjects/models_and_mcts/mcts_results/baisline_16.12_Q_softmax.csv')
    df_random = pd.read_csv('/home/mayatarno/PycharmProjects/models_and_mcts/mcts_results/baisline_16.12_random.csv')
    """
    df_mcts_softmax = pd.read_csv('/home/mayatarno/PycharmProjects/models_and_mcts/mcts_results/mcts_16.12_softmax_policy.csv')
    print('df_mcts_softmax',sum(df_mcts_softmax['total_payoff'].values)/float(len(df_mcts_softmax['total_payoff'].values)))
    """
    print('df_dm_hard',sum(df_dm_hard['total_payoff'].values)/float(len(df_dm_hard['total_payoff'].values)))
    print('df_dm_softmax',sum(df_dm_softmax['total_payoff'].values)/float(len(df_dm_softmax['total_payoff'].values)))
    print('df_only3',sum(df_only3['total_payoff'].values)/float(len(df_only3['total_payoff'].values)))
    print('df_only6',sum(df_only6['total_payoff'].values)/float(len(df_only6['total_payoff'].values)))
    print('df_q_hard',sum(df_q_hard['total_payoff'].values)/float(len(df_q_hard['total_payoff'].values)))
    print('df_q_softmax',sum(df_q_softmax['total_payoff'].values)/float(len(df_q_softmax['total_payoff'].values)))
    print('df_random',sum(df_random['total_payoff'].values)/float(len(df_random['total_payoff'].values)))



    #f7()
    # df_q = pd.read_csv('/home/mayatarno/PycharmProjects/models_and_mcts/mcts_results/sum_q_baisline11.11.csv')
    # df_dm = pd.read_csv('/home/mayatarno/PycharmProjects/models_and_mcts/mcts_results/sum_dm_baisline11.11.csv')
    # real_user = pd.read_csv('/home/mayatarno/PycharmProjects/models_and_mcts/mcts_results/sum_realexpert_baisline11.11.csv')
    # only6 = pd.read_csv('/home/mayatarno/PycharmProjects/models_and_mcts/mcts_results/sum_only6_11.11.csv')
    # mcts_norollout03 = pd.read_csv('/home/mayatarno/PycharmProjects/models_and_mcts/mcts_results/sum_mcts_11.11.csv')
    # sum_dm_q_baisline = pd.read_csv('/home/mayatarno/PycharmProjects/models_and_mcts/mcts_results/sum_dm+q_baisline11.11.csv')
    # mcts_norollout05 = pd.read_csv('/home/mayatarno/PycharmProjects/models_and_mcts/mcts_results/sum_mcts0.5_11.11.csv')
    # mcts_norollout02 = pd.read_csv('/home/mayatarno/PycharmProjects/models_and_mcts/mcts_results/sum_mcts0.2_11.11.csv')
    # mcts_05_all_q = pd.read_csv('/home/mayatarno/PycharmProjects/models_and_mcts/mcts_results/sum_mcts_with_rollout0.5_15.11.csv')
    # mcts_05_nol_q = pd.read_csv('/home/mayatarno/PycharmProjects/models_and_mcts/mcts_results/sum_mcts_without_Q.csv')
    # print('mcts_05_all_q',sum(mcts_05_all_q['total_payoff'].values)/float(len(mcts_05_all_q['total_payoff'].values)))
    # print('mcts_05_nol_q',sum(mcts_05_nol_q['total_payoff'].values)/float(len(mcts_05_nol_q['total_payoff'].values)))
    #
    # #real_user=real_user[real_user.index<=185]
    # print('df_dm',sum(df_dm['total_payoff'].values)/float(len(df_dm['total_payoff'].values)))
    # print('mcts_norollout0.3',sum(mcts_norollout03['total_payoff'].values)/float(len(mcts_norollout03['total_payoff'].values)))
    # print('mcts_norollout0.2',sum(mcts_norollout02['total_payoff'].values)/float(len(mcts_norollout02['total_payoff'].values)))
    # print('mcts_norollout0.5',sum(mcts_norollout05['total_payoff'].values)/float(len(mcts_norollout05['total_payoff'].values)))
    # print('only6',sum(only6['total_payoff'].values)/float(len(only6['total_payoff'].values)))
    # print('df_q',sum(df_q['total_payoff'].values) / float(len(df_q['total_payoff'].values)))
    # print('real_user',sum(real_user['total_payoff'].values) / float(len(real_user['total_payoff'].values)))
    # #print('real_user2',sum(real_user2['total_payoff'].values) / float(len(real_user2['total_payoff'].values)))
    # print('sum_dm_q_baisline',sum(sum_dm_q_baisline['total_payoff'].values) / float(len(sum_dm_q_baisline['total_payoff'].values)))
    #
    # mcts_norollout02=mcts_norollout02[mcts_norollout02.index<=548]
    # mcts_norollout03=mcts_norollout03[mcts_norollout03.index<=548]
    # mcts_norollout05=mcts_norollout05[mcts_norollout05.index<=548]
    # df_dm=df_dm[df_dm.index<=548]
    # #real_user=real_user[real_user.index<=132]
    # print(len(real_user))
    # print(len(mcts_norollout05))
    # #new_df = df6(real_user)
    # print('v')
    # #f3(df_dm,mcts_norollout05,real_user)
    # f5(mcts_norollout05,real_user)
    """