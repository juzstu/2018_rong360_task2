# -*- coding：utf-8 -*-
# Author: Juzphy
# Time: 2018/10/8 21:30
import pandas as pd
from collections import defaultdict
import numpy as np
import os
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale
import time
from sklearn.metrics import roc_auc_score
from scipy import stats
import warnings
from sklearn.linear_model import LinearRegression


warnings.filterwarnings('ignore')

path = './rong360/open_data'
origin_train = pd.read_csv(path + '/sample_train.txt', encoding='utf8', sep='\t')
origin_val = pd.read_csv(path + '/valid_id.txt', encoding='utf8')
origin_test = pd.read_csv(path + '/test_id.txt', encoding='utf8')
app_path = path + '/dat_app'
edge_path = path + './dat_edge'

if not os.path.exists(path + '/mid_data'):
    os.mkdir(path + '/mid_data')


def data_app_deal():
    app_data = pd.read_csv(app_path + '/dat_app_1', encoding='utf8', sep='\t', names=['id', 'app_ids'])
    print('Now, app shape: ', app_data.shape)
    for i in range(2, 8):
        tmp = pd.read_csv(app_path + f'/dat_app_{i}', encoding='utf8', sep='\t', names=['id', 'app_ids'])
        app_data = pd.concat([app_data, tmp], axis=0).reset_index(drop=True)
        print('Now, app shape: ', app_data.shape)
        del tmp
    app_data.to_csv(path + '/mid_data/dat_app_all.csv', index=False)


def merge_app(data_frame, mode):
    all_app = pd.read_csv(path + '/mid_data/dat_app_all.csv', encoding='utf8')
    tmp_merge = pd.merge(data_frame, all_app, on=['id'], how='left')
    tmp_merge.to_csv(path + f'/mid_data/{mode}_merge_app.csv', encoding='utf8', index=False)


def data_edge_deal(data_frame, mode, is_to_id=True):
    edge = pd.read_csv(edge_path + '/dat_edge_1', encoding='utf8', sep='\t')
    if is_to_id:
        edge = edge[edge['to_id'].isin(data_frame['id'])]
    else:
        edge = edge[edge['from_id'].isin(data_frame['id'])]
    print('Now, edge shape: ', edge.shape)
    for i in range(2, 12):
        tmp = pd.read_csv(edge_path + f'/dat_edge_{i}', encoding='utf8', sep='\t', names=['from_id', 'to_id', 'info'])
        if is_to_id:
            tmp = tmp[tmp['to_id'].isin(data_frame['id'])]
        else:
            tmp = tmp[tmp['from_id'].isin(data_frame['id'])]
        edge = pd.concat([edge, tmp], axis=0).reset_index(drop=True)
        print('Now, edge shape: ', edge.shape)
        del tmp
    edge.to_csv(path + f'/mid_data/dat_edge_{mode}_{int(is_to_id)}.csv', index=False)


# app_idx 计数特征， 前635 One Hot
def stat_top_app():
    all_app = pd.read_csv(path + '/mid_data/dat_app_all.csv', encoding='utf8')
    app_dict = defaultdict(int)
    for i in all_app['app_ids'].values:
        if isinstance(i, str):
            for d in i.split(','):
                app_dict[d] += 1
    tmp = pd.DataFrame()
    tmp['app_idx'] = app_dict.keys()
    tmp['app_cnt'] = app_dict.values()
    tmp.sort_values(by='app_cnt', ascending=False, inplace=True)
    tmp.to_csv(path + '/mid_data/app_stat.csv', encoding='utf8', index=False)


def get_date(x, desc, index):
    if desc in x:
        tmp = x.split(desc + ':')[1].split('_')
        if ',' not in tmp[index]:
            return float(tmp[index])
        else:
            return float(tmp[index].split(',')[0])
    else:
        return np.nan


def q10(x):
    return x.quantile(0.1)


def q20(x):
    return x.quantile(0.2)


def q30(x):
    return x.quantile(0.3)


def q40(x):
    return x.quantile(0.4)


def q60(x):
    return x.quantile(0.6)


def q70(x):
    return x.quantile(0.7)


def q80(x):
    return x.quantile(0.8)


def q90(x):
    return x.quantile(0.9)


def kurt_apply(x):
    return stats.kurtosis(x)


def mode_num(x):
    return stats.mode(x)[0][0]


def mode_count(x):
    return stats.mode(x)[1][0]


def mode_multi(x):
    return stats.mode(x)[1][0] * stats.mode(x)[0][0]


def date_stat(data_frame, date_list, mode):
    tmp = data_frame[date_list]
    data_frame[f"date_{mode}_mean"] = tmp.mean(axis=1)
    data_frame[f"date_{mode}_max-min"] = tmp.max(axis=1) - tmp.min(axis=1)
    data_frame[f"date_{mode}_median"] = tmp.median(axis=1)
    data_frame[f"date_{mode}_min"] = tmp.min(axis=1)
    data_frame[f"date_{mode}_max"] = tmp.max(axis=1)


# 将id对应的cnt和weight做线性回归，得出拟和的相应系数和截距
# 该特征群没有提升
def linear(df):
    x, y = df.split('$')
    x, y = np.fromstring(x.replace('[', '').replace(']', ''), dtype=np.float, sep=' ').reshape(-1, 1), \
           np.fromstring(y.replace('[', '').replace(']', ''), dtype=np.float, sep=' ').reshape(-1, 1)
    if len(x) > 2:
        lr = LinearRegression()
        lr.fit(x+10, y+10)
        return str(lr.coef_.flatten()[0]) + '$' + str(lr.intercept_[0])
    else:
        return 'nan'


# edge 统计特征
def edge_stat(data_frame, x, y, merge_data, mode):
    print(f'START STAT EDGE FEATURES of {mode}...')
    data_frame['2017-11_cnt'] = data_frame['info'].apply(get_date, args=('2017-11', 0))
    data_frame['2017-11_weight'] = data_frame['info'].apply(get_date, args=('2017-11', 1))
    data_frame['2017-11_cw'] = data_frame['2017-11_cnt'] * data_frame['2017-11_weight']

    data_frame['2017-12_cnt'] = data_frame['info'].apply(get_date, args=('2017-12', 0))
    data_frame['2017-12_weight'] = data_frame['info'].apply(get_date, args=('2017-12', 1))
    data_frame['2017-12_cw'] = data_frame['2017-12_cnt'] * data_frame['2017-12_weight']

    data_frame['2018-01_cnt'] = data_frame['info'].apply(get_date, args=('2018-01', 0))
    data_frame['2018-01_weight'] = data_frame['info'].apply(get_date, args=('2018-01', 1))
    data_frame['2018-01_cw'] = data_frame['2018-01_cnt'] * data_frame['2018-01_weight']

    date_stat(data_frame, ['2017-11_cnt', '2017-12_cnt', '2018-01_cnt'], 'cnt')
    date_stat(data_frame, ['2017-11_weight', '2017-12_weight', '2018-01_weight'], 'weight')
    date_stat(data_frame, ['2017-11_cw', '2017-12_cw', '2018-01_cw'], 'cw')

    data_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    date_stat_list = []
    for i in ['cnt', 'weight', 'cw']:
        date_stat_list.extend([f"date_{i}_mean",  f"date_{i}_max-min",
                               f"date_{i}_min", f"date_{i}_max", f"date_{i}_median"
                               ])
    tmp_unique = data_frame.groupby(by=x)[y].agg(['nunique']).reset_index()
    tmp_unique.columns = ['id', x + '_nunique']
    merge_data = pd.merge(merge_data, tmp_unique, how='left', on='id')

    date_features = ['2017-11_cnt', '2017-11_weight', '2017-11_cw',
                     '2017-12_cnt', '2017-12_weight', '2017-12_cw',
                     '2018-01_cnt', '2018-01_weight', '2018-01_cw'
                     ]

    # 将id对应的cnt和weight转成散点图形式
    date_pairs = [('2017-11_cnt', '2017-11_weight'), ('2017-12_cnt', '2017-12_weight'), ('2018-01_cnt', '2018-01_weight')]
    for dp in date_pairs:
        cnt_tmp = data_frame.groupby(x)[dp[0]].agg(lambda n: [0] if n.isnull().all() else
                        list(n.dropna())).reset_index()
        cnt_tmp.columns = ['id', f'{x}_{dp[0]}_list']
        cnt_tmp[f'{x}_{dp[0]}_list'] = cnt_tmp[f'{x}_{dp[0]}_list'].apply(scale)
        cnt_tmp[f'{x}_{dp[0]}_min'] = cnt_tmp[f'{x}_{dp[0]}_list'].apply(np.min) + 10
        cnt_tmp[f'{x}_{dp[0]}_max'] = cnt_tmp[f'{x}_{dp[0]}_list'].apply(np.max) + 10
        cnt_tmp[f'{x}_{dp[0]}_freq'] = cnt_tmp[f'{x}_{dp[0]}_list'].apply(lambda h: stats.mode(h)[0][0]+10)

        weight_tmp = data_frame.groupby(x)[dp[1]].agg(lambda n: [0] if n.isnull().all() else list(n.dropna())).reset_index()
        weight_tmp.columns = ['id', f'{x}_{dp[1]}_list']
        weight_tmp[f'{x}_{dp[1]}_list'] = weight_tmp[f'{x}_{dp[1]}_list'].apply(scale)
        weight_tmp[f'{x}_{dp[1]}_min'] = weight_tmp[f'{x}_{dp[1]}_list'].apply(np.min) + 10
        weight_tmp[f'{x}_{dp[1]}_max'] = weight_tmp[f'{x}_{dp[1]}_list'].apply(np.max) + 10
        weight_tmp[f'{x}_{dp[1]}_freq'] = weight_tmp[f'{x}_{dp[1]}_list'].apply(lambda h: stats.mode(h)[0][0]+10)
        cw_tmp = pd.merge(cnt_tmp, weight_tmp, on='id')
        cw_tmp[f'{x}_{dp[0][:-4]}_ld'] = np.sqrt(cw_tmp[f'{x}_{dp[0]}_min'] ** 2 + cw_tmp[f'{x}_{dp[1]}_min'] ** 2)
        cw_tmp[f'{x}_{dp[0][:-4]}_lu'] = np.sqrt(cw_tmp[f'{x}_{dp[0]}_min'] ** 2 + cw_tmp[f'{x}_{dp[1]}_max'] ** 2)
        cw_tmp[f'{x}_{dp[0][:-4]}_rd'] = np.sqrt(cw_tmp[f'{x}_{dp[0]}_max'] ** 2 + cw_tmp[f'{x}_{dp[1]}_min'] ** 2)
        cw_tmp[f'{x}_{dp[0][:-4]}_ru'] = np.sqrt(cw_tmp[f'{x}_{dp[0]}_max'] ** 2 + cw_tmp[f'{x}_{dp[1]}_max'] ** 2)
        cw_tmp[f'{x}_{dp[0][:-4]}_mid'] = np.sqrt(((cw_tmp[f'{x}_{dp[0]}_max'] - cw_tmp[f'{x}_{dp[0]}_min']) / 2 +
                                                 cw_tmp[f'{x}_{dp[0]}_min']) ** 2 +
                                                ((cw_tmp[f'{x}_{dp[1]}_max'] - cw_tmp[f'{x}_{dp[1]}_min']) / 2 +
                                                 cw_tmp[f'{x}_{dp[1]}_min']) ** 2)
        cw_tmp[f'{x}_{dp[0][:-4]}_freq'] = np.sqrt(cw_tmp[f'{x}_{dp[0]}_freq'] ** 2 + cw_tmp[f'{x}_{dp[1]}_freq'] ** 2)

        cw_tmp[f'{x}_{dp[0][:-4]}_length'] = cw_tmp[f'{x}_{dp[0]}_max'] - cw_tmp[f'{x}_{dp[0]}_min']
        cw_tmp[f'{x}_{dp[0][:-4]}_width'] = cw_tmp[f'{x}_{dp[1]}_max'] - cw_tmp[f'{x}_{dp[1]}_min']
        cw_tmp[f'{x}_{dp[0][:-4]}_square'] = cw_tmp[f'{x}_{dp[0][:-4]}_length'] * cw_tmp[f'{x}_{dp[0][:-4]}_width']
        cw_tmp[f'{x}_{dp[0][:-4]}_line'] = np.sqrt((cw_tmp[f'{x}_{dp[0]}_max'] - cw_tmp[f'{x}_{dp[0]}_min']) ** 2 +
                                                   (cw_tmp[f'{x}_{dp[1]}_max'] - cw_tmp[f'{x}_{dp[1]}_min']) ** 2)

        # cw_tmp[f'{x}_{dp[0][:-4]}_mean'] = np.sqrt(cw_tmp[f'{x}_{dp[1]}_list'].apply(np.mean) ** 2 +
        #                                            cw_tmp[f'{x}_{dp[0]}_list'].apply(np.mean) ** 2)
        #
        # cw_tmp[f'{x}_{dp[0][:-4]}_median'] = np.sqrt(cw_tmp[f'{x}_{dp[1]}_list'].apply(np.median) ** 2 +
        #                                            cw_tmp[f'{x}_{dp[0]}_list'].apply(np.median) ** 2)
        # cw_tmp[f'{x}_{dp[0][:-4]}_xy'] = cnt_tmp[f'{x}_{dp[0]}_list'].astype(str) + '$' +
        # weight_tmp[f'{x}_{dp[1]}_list'].astype(str)
        # cw_tmp[f'{x}_{dp[0][:-4]}_lr'] = cw_tmp[f'{x}_{dp[0][:-4]}_xy'].apply(linear)
        # cw_tmp[f'{x}_{dp[0][:-4]}_coef'] = cw_tmp[f'{x}_{dp[0][:-4]}_lr'].apply(
        #     lambda s: np.nan if s == 'nan' else float(s.split('$')[0])*10)
        # cw_tmp[f'{x}_{dp[0][:-4]}_intercept'] = cw_tmp[f'{x}_{dp[0][:-4]}_lr'].apply(
        #     lambda s: np.nan if s == 'nan' else float(s.split('$')[1]))

        tmp_list = ['id', f'{x}_{dp[0][:-4]}_ld', f'{x}_{dp[0][:-4]}_lu', f'{x}_{dp[0][:-4]}_rd', f'{x}_{dp[0][:-4]}_ru',
                    f'{x}_{dp[0][:-4]}_mid', f'{x}_{dp[0][:-4]}_freq', f'{x}_{dp[0][:-4]}_length',
                    f'{x}_{dp[0][:-4]}_width', f'{x}_{dp[0][:-4]}_square', f'{x}_{dp[0][:-4]}_line']
                    # f'{x}_{dp[0][:-4]}_mean', f'{x}_{dp[0][:-4]}_median', f'{x}_{dp[0][:-4]}_coef',
                    # f'{x}_{dp[0][:-4]}_intercept']
        merge_data = pd.merge(merge_data, cw_tmp[tmp_list], how='left', on='id')
        del cnt_tmp, weight_tmp, cw_tmp
    date_features.extend(date_stat_list)

    stat_list = ['mean', 'std', 'skew', 'mad', 'median', 'count', 'max', 'min', 'var', kurt_apply, mode_num,
                 mode_count, mode_multi, q10, q20, q30, q40, q60, q70, q80, q90]

    for d in date_features:
        print(f'Now deal the {d} of {x} ...')
        date_tmp = data_frame.groupby(x)[d].agg(stat_list).reset_index()
        dcols = [x + '_' + d + '_' + c for c in stat_list[:9]]
        q_list = [x + '_' + d + '_' + c for c in ['kurt', 'mode_num', 'mode_count', 'model_multi', 'q_10',
                                                  'q_20', 'q_30', 'q_40', 'q_60', 'q_70', 'q_80', 'q_90', ]]
        dcols.extend(q_list)
        id_dcols = ['id'] + dcols
        date_tmp.columns = id_dcols
        merge_data = pd.merge(merge_data, date_tmp, how='left', on='id')

    return merge_data


def edge_merge(data_frame, mode):
    from_id = pd.read_csv(path + f'/mid_data/dat_edge_{mode}_0.csv')
    to_id = pd.read_csv(path + f'/mid_data/dat_edge_{mode}_1.csv')
    from_data = edge_stat(from_id, 'from_id', 'to_id', data_frame, mode)
    to_data = edge_stat(to_id, 'to_id', 'from_id', from_data, mode)

    print(f'{mode} shape is: {to_data.shape}')
    to_data.to_csv(path + f'/mid_data/{mode}_merge_edge.csv', encoding='utf8', index=False)


def gen_one_hot(df, desc):
    if isinstance(df, list):
        if desc in df:
            return 1
        else:
            return 0
    else:
        return np.nan


def is_top_app(df, desc):
    if isinstance(df, str):
        tmp = [i in desc for i in df.split(',')]
        if np.all(tmp):
            return 0
        else:
            return 1
    else:
        return np.nan


def gen_sym_data():
    sym = pd.read_csv(path + '/dat_symbol.txt', encoding='utf8', sep='\t')
    sym['sym_len'] = sym['symbol'].apply(lambda x: len(x.split(',')))
    sym_all = []
    for i in sym['symbol'].values:
        for j in i.split(','):
            if j not in sym_all:
                sym_all.append(j)

    for s in sym_all:
        sym[s] = sym['symbol'].apply(gen_one_hot, args=(s,))
    sym.drop(['symbol'], axis=1, inplace=True)
    sym.to_csv(path + f'/mid_data/all_symbol.csv', encoding='utf8', index=False)


def gen_app_data(mode, threshold):
    data_frame = pd.read_csv(path + f'/mid_data/{mode}_merge_app.csv', encoding='utf8')
    top_app = pd.read_csv(path + f'/mid_data/app_stat.csv')['app_ids'].values[:threshold]
    top_app = [str(t) for t in top_app]
    data_frame['app_ids'] = data_frame['app_ids'].astype(str)
    data_frame['app_ids'] = data_frame['app_ids'].apply(lambda x: x.split(',') if x != 'nan' else np.nan)
    data_frame['app_len'] = data_frame['app_ids'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    for a in top_app:
        data_frame[a] = data_frame['app_ids'].apply(gen_one_hot, args=(a,))
    if mode == 'train':
        data_frame.drop(['app_ids', 'label'], axis=1, inplace=True)
    else:
        data_frame.drop(['app_ids'], axis=1, inplace=True)
    print(f'{mode} shape is: {data_frame.shape}')
    data_frame.to_csv(path + f'/mid_data/{mode}_app_feat.csv', encoding='utf8', index=False)


# 该特征群没有提升
def gen_avg_app(mode):
    data_frame = pd.read_csv(path + f'/mid_data/{mode}_merge_app.csv', encoding='utf8')
    top_app = pd.read_csv(path + f'/mid_data/app_stat.csv')['app_ids']
    top_app_dict = dict(zip(top_app.values, top_app.index))
    data_frame['app_ids'] = data_frame['app_ids'].astype(str)
    data_frame['app_ids'] = data_frame['app_ids'].apply(lambda x: x.split(',') if x != 'nan' else np.nan)
    data_frame['app_len'] = data_frame['app_ids'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    data_frame['app_ids'] = data_frame['app_ids'].apply(lambda x: np.array([top_app_dict[int(i)] for i in x]) if isinstance(x, list) else np.nan)
    data_frame['app_sort_total'] = data_frame['app_ids'].apply(lambda x: np.sum(x))
    data_frame['app_avg'] = data_frame['app_ids'].apply(lambda x: np.mean(x))
    data_frame['app_max'] = data_frame['app_ids'].apply(lambda x: np.max(x))
    data_frame['app_min'] = data_frame['app_ids'].apply(lambda x: np.min(x))
    data_frame['app_median'] = data_frame['app_ids'].apply(lambda x: np.median(x))
    data_frame['app_pert_1'] = data_frame['app_ids'].apply(lambda x: np.percentile(x, 10))
    data_frame['app_pert_2'] = data_frame['app_ids'].apply(lambda x: np.percentile(x, 20))
    data_frame['app_pert_3'] = data_frame['app_ids'].apply(lambda x: np.percentile(x, 30))
    data_frame['app_pert_4'] = data_frame['app_ids'].apply(lambda x: np.percentile(x, 40))
    data_frame['app_pert_6'] = data_frame['app_ids'].apply(lambda x: np.percentile(x, 60))
    data_frame['app_pert_7'] = data_frame['app_ids'].apply(lambda x: np.percentile(x, 70))
    data_frame['app_pert_8'] = data_frame['app_ids'].apply(lambda x: np.percentile(x, 80))
    data_frame['app_pert_9'] = data_frame['app_ids'].apply(lambda x: np.percentile(x, 90))
    data_frame['app_pert_std'] = data_frame['app_ids'].apply(lambda x: np.std(x))
    data_frame['app_pert_var'] = data_frame['app_ids'].apply(lambda x: np.var(x))
    data_frame['app_pert_skew'] = data_frame['app_ids'].apply(lambda x: stats.skew(x))
    data_frame['app_pert_kurt'] = data_frame['app_ids'].apply(lambda x: stats.kurtosis(x))
    data_frame['app_max-min'] = data_frame['app_max'] - data_frame['app_min']
    data_frame['app_scale'] = (data_frame['app_max'] - data_frame['app_avg'])/(data_frame['app_avg'] - data_frame['app_min'])
    if mode == 'train':
        data_frame.drop(['app_ids', 'label'], axis=1, inplace=True)
    else:
        data_frame.drop(['app_ids'], axis=1, inplace=True)
    print(f'{mode} shape is: {data_frame.shape}')
    data_frame.to_csv(path + f'/mid_data/{mode}_app_sort.csv', encoding='utf8', index=False)


# 该特征群没有提升
def gen_id_cnt(mode, is_to_id=True):
    if is_to_id:
        label = 'to_id'
    else:
        label = 'from_id'
    data_frame = pd.read_csv(path + f'/mid_data/dat_edge_{mode}_{int(is_to_id)}.csv')
    file = open(path + f'/mid_data/dat_edge_{mode}_{int(is_to_id)}.txt', 'w')
    file.write('id,date\n')
    for i in range(data_frame.shape[0]):
        info_tmp = data_frame.iloc[i]['info'].split(',')
        print(i)
        for it in info_tmp:
            file.write(f'{data_frame.iloc[i][label]},{it[:7]}\n')
    file.close()
    data_frame_tm = pd.read_csv(path + f'/mid_data/dat_edge_{mode}_{int(is_to_id)}.txt')
    data_frame_tm = list(data_frame_tm.groupby(by=['date'])['id'])
    tmp_date = None
    for j in data_frame_tm:
        j_tmp = pd.DataFrame(j[1].value_counts().values, columns=[f'{j[0]}_{label}_cnt'])
        j_tmp['id'] = j[1].value_counts().index
        if isinstance(tmp_date, pd.core.frame.DataFrame):
            tmp_date = pd.merge(tmp_date, j_tmp, on='id', how='outer')
        else:
            tmp_date = j_tmp
    tmp_date.fillna(0, inplace=True)
    tmp_date[f'dt_sum_{label}'] = tmp_date[f'2017-11_{label}_cnt'] + tmp_date[f'2017-12_{label}_cnt'] + \
                                  tmp_date[f'2018-01_{label}_cnt']
    print(f'{mode} of {label} shape: {tmp_date.shape}')
    tmp_date.to_csv(path + f'/mid_data/{mode}_{label}_cnt.csv', encoding='utf8', index=False)


def id_cnt_merge():
    gen_id_cnt('val', is_to_id=False)
    gen_id_cnt('val', is_to_id=True)
    gen_id_cnt('train', is_to_id=False)
    gen_id_cnt('train', is_to_id=True)


def final_data_merge(mode):
    edge_frame = pd.read_csv(path + f'/mid_data/{mode}_merge_edge.csv', encoding='utf8')
    app_frame_oh = pd.read_csv(path + f'/mid_data/{mode}_app_feat.csv', encoding='utf8')
    sym = pd.read_csv(path + f'/mid_data/all_symbol.csv', encoding='utf8')
    origin_risk = pd.read_csv(path + '/dat_risk.txt', encoding='utf8', sep='\t')
    # 队友wzm 根据id做的 tf-idf特征
    wzm = pd.read_csv(path + f'/mid_data/wzm_{mode}_tf_idf.csv')

    data_frame = pd.merge(edge_frame, app_frame_oh, how='left', on='id')
    data_frame = pd.merge(data_frame, sym, how='left', on='id')
    data_frame = pd.merge(data_frame, origin_risk, how='left', on='id')
    data_frame = pd.merge(data_frame, wzm, how='left', on='id')
    if not os.path.exists(path + '/final_data'):
        os.mkdir(path + '/final_data')
    print(f'{mode} shape is: {data_frame.shape}')
    data_frame.to_csv(path + f'/final_data/{mode}_data_final.csv', encoding='utf8', index=False)


def lgb_train(train_, test_, feature_list):
    print(f'Start train lgb model, file describe: ', train_.shape, test_.shape)
    params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 62,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 3,
        'feature_fraction_seed': 2,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'nthread': -1
    }
    n_split = 5
    X, y = train_[feature_list], train_['label']
    skf = StratifiedKFold(n_splits=n_split, random_state=1024, shuffle=True)
    roc_list = []
    sub_prob = np.zeros((test_.shape[0], n_split))
    feat_imp = pd.DataFrame()
    feat_imp['features'] = feature_list
    for index, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        print(f'the {index} training start ...')
        X_train, X_valid, y_train, y_valid = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        print(X_train.shape, X_valid.shape)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)
        gbm = lgb.train(params, train_data,
                        num_boost_round=5000,
                        valid_sets=valid_data,
                        early_stopping_rounds=100,
                        verbose_eval=100)
        feat_imp[f'Fold_{index}_imp'] = gbm.feature_importance()
        x_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
        roc_list.append(roc_auc_score(y_valid, x_pred))
        sub_prob[:, index - 1] = gbm.predict(test_[feature_list], num_iteration=gbm.best_iteration)
    feat_imp['imp_mean'] = (feat_imp[f'Fold_1_imp'] + feat_imp[f'Fold_2_imp'] + feat_imp[f'Fold_3_imp'] +
    feat_imp[f'Fold_4_imp'] + feat_imp[f'Fold_5_imp']) / 5
    if not os.path.exists(path + '/feat'):
        os.mkdir(path + '/feat')
    feat_imp.to_csv(path + '/feat/5_fold_feat.csv', index=False, encoding='gbk')
    test_['prob'] = np.mean(sub_prob, axis=1)
    date = time.strftime('%Y%m%d_%H%M%S')
    if not os.path.exists(path + '/result'):
        os.mkdir(path + '/result')
    test_[['id', 'prob']].to_csv(path + f'/result/lgb_task2_mean_{date}.txt', encoding='utf8', index=False)
    print('submit file successful generate ...')
    print(f'{n_split} folds: {roc_list}, avg: {np.mean(roc_list)}')


def xgb_train(train_, test_, feature_list):
    print(f'Start train lgb model, file describe: ', train_.shape, test_.shape)
    n_splits = 5

    xgb_params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'gamma': 0.1,
        'min_child_weight': 1.1,
        'max_depth': 6,
        # 'alpha': 5,
        'lambda': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.7,
        'eta': 0.01,
        'tree_method': 'exact',
        'seed': 2018,
        'silent': 1
    }
    X, y = train_[feature_list], train_['label']
    skf = StratifiedKFold(n_splits=n_splits, random_state=1024, shuffle=True)
    roc_list = []
    sub_prob = np.zeros((test_.shape[0], n_splits))
    for index, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        print(f'the {index} training start ...')
        X_train, X_valid, y_train, y_valid = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        print(X_train.shape, X_valid.shape)
        train_data = xgb.DMatrix(X_train, label=y_train, feature_names=feature_list)
        valid_data = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_list)
        watchlist = [(valid_data, 'val')]
        xgb_val = xgb.train(xgb_params, train_data,
                            num_boost_round=5000,
                            evals=watchlist,
                            early_stopping_rounds=100,
                            verbose_eval=100)

        x_pred = xgb_val.predict(valid_data, ntree_limit=xgb_val.best_iteration)
        roc_list.append(roc_auc_score(y_valid, x_pred))

        test_data = xgb.DMatrix(test_[feature_list], feature_names=feature_list)
        sub_prob[:, index-1] = xgb_val.predict(test_data, ntree_limit=xgb_val.best_iteration)

    test_['prob'] = np.mean(sub_prob, axis=1)
    date = time.strftime('%Y%m%d_%H%M%S')
    if not os.path.exists(path + '/result'):
        os.mkdir(path + '/result')
    test_[['id', 'prob']].to_csv(path + f'/result/xgb_task2_{date}.txt', encoding='utf8', index=False)
    print('submit file successful generate ...')
    print(f'{n_splits} folds: {roc_list}, avg: {np.mean(roc_list)}')


if __name__ == "__main__":
    data_app_deal()
    merge_app(origin_test, 'test')
    merge_app(origin_train, 'train')
    data_edge_deal(origin_train, 'train', is_to_id=False)
    data_edge_deal(origin_train, 'train', is_to_id=True)
    data_edge_deal(origin_test, 'test', is_to_id=False)
    data_edge_deal(origin_test, 'test', is_to_id=True)
    gen_sym_data()
    gen_app_data('test', 635)
    gen_app_data('train', 635)
    edge_merge(origin_test, 'test')
    edge_merge(origin_train, 'train')
    final_data_merge('train')
    final_data_merge('test')
    train = pd.read_csv(path + f'/final_data/train_data_final.csv', encoding='utf8')
    val = pd.read_csv(path + f'/final_data/test_data_final.csv', encoding='utf8')
    features = [f for f in val.columns if f not in ['id']]
    xgb_train(train, val, features)
