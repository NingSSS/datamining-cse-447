import numpy as np
import pandas as pd
import re as re
import warnings
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb


from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_csv('train_data.csv')
label = pd.read_csv('train_label.csv')
tdata = pd.read_csv('test_data.csv')
result = pd.read_csv('submission.csv')

data['type'].fillna('international',inplace = True)
tdata['type'].fillna('international',inplace = True)
# data['lvl3'].fillna('unique',inplace = True)
# tdata['lvl3'].fillna('unique',inplace = True)
# # data['lvl2']=data['lvl2'].str.cat(data['lvl1'],sep='_')
# data['lvl3']=data['lvl3'].str.cat(data['lvl2'],sep='_')
# # tdata['lvl2']=tdata['lvl2'].str.cat(tdata['lvl1'],sep='_')
# tdata['lvl3']=tdata['lvl3'].str.cat(tdata['lvl2'],sep='_')
#data.loc[data['lvl3'].isnull(),'lvl3']=data[data['lvl3'].isnull()]['lvl2']

data['d_quality'] = data['descrption'].str.extract('(Quality|quality)', expand=False)
data['d_quality'].fillna('normal',inplace = True)
data.loc[(data['d_quality']=='Quality') | (data['d_quality']=='quality'),'quality']='quality'
tdata['d_quality'] = tdata['descrption'].str.extract('(Quality|quality)', expand=False)
tdata['d_quality'].fillna('normal',inplace = True)
tdata.loc[(tdata['d_quality']=='Quality') | (tdata['d_quality']=='quality'),'quality']='quality'

data['d_new'] = data['descrption'].str.extract('(100%)', expand=False)
data['d_new'].fillna('normal',inplace = True)
tdata['d_new'] = tdata['descrption'].str.extract('(100%)', expand=False)
tdata['d_new'].fillna('normal',inplace = True)
# data['fashion'] = data['name'].str.extract('(Fashion|fashion)', expand=False)
# data['fashion'].fillna('unfashion',inplace = True)
# data.loc[(data['fashion']=='Fashion') | (data['fashion']=='fashion'),'fashion']='fashion'
# tdata['fashion'] = tdata['name'].str.extract('(Fashion|fashion)', expand=False)
# tdata['fashion'].fillna('unfashion',inplace = True)
# tdata.loc[(tdata['fashion']=='Fashion') | (tdata['fashion']=='fashion'),'fashion']='fashion'
# # data['iPhone'] = data['descrption'].str.extract('(iPhone)', expand=False)
# # data['iPhone'].fillna('normal',inplace = True)
# # tdata['iPhone'] = tdata['descrption'].str.extract('(iPhone)', expand=False)
# # tdata['iPhone'].fillna('normal',inplace = True)
# data['durable'] = data['descrption'].str.extract('(durable)', expand=False)
# data['durable'].fillna('normal',inplace = True)
# tdata['durable'] = tdata['descrption'].str.extract('(durable)', expand=False)
# tdata['durable'].fillna('normal',inplace = True)
# # data['health'] = data['descrption'].str.extract('(Health|health)', expand=False)
# # data['health'].fillna('normal',inplace = True)
# # data.loc[(data['health']=='health') | (data['health']=='Health'),'health']='health'
# # tdata['health'] = tdata['descrption'].str.extract('(Health|health)', expand=False)
# # tdata['health'].fillna('normal',inplace = True)
# # tdata.loc[(tdata['health']=='health') | (tdata['health']=='Health'),'health']='health'
# # data['beauty'] = data['descrption'].str.extract('(Beauty|beauty)', expand=False)
# # data['beauty'].fillna('normal',inplace = True)
# # data.loc[(data['beauty']=='beauty') | (data['beauty']=='Beauty'),'beauty']='beauty'
# # tdata['beauty'] = tdata['descrption'].str.extract('(Beauty|beauty)', expand=False)
# # tdata['beauty'].fillna('normal',inplace = True)
# # tdata.loc[(tdata['beauty']=='beauty') | (tdata['beauty']=='Beauty'),'beauty']='beauty'
# # data['easy'] = data['descrption'].str.extract('(easy)', expand=False)
# # data['easy'].fillna('normal',inplace = True)
# # tdata['easy'] = tdata['descrption'].str.extract('(easy)', expand=False)
# # tdata['easy'].fillna('normal',inplace = True)
# data['color'] = data['descrption'].str.extract('(Color)', expand=False)
# data['color'].fillna('normal',inplace = True)
# tdata['color'] = tdata['descrption'].str.extract('(Color)', expand=False)
# tdata['color'].fillna('normal',inplace = True)
# data['power'] = data['descrption'].str.extract('(Power)', expand=False)
# data['power'].fillna('normal',inplace = True)
# tdata['power'] = tdata['descrption'].str.extract('(Power)', expand=False)
# tdata['power'].fillna('normal',inplace = True)
# data['style'] = data['descrption'].str.extract('(Style)', expand=False)
# data['style'].fillna('normal',inplace = True)
# tdata['style'] = tdata['descrption'].str.extract('(Style)', expand=False)
# tdata['style'].fillna('normal',inplace = True)
# #     tdata['durable'] = tdata['descrption'].str.extract('(durable)', expand=False)
# #     tdata['durable'].fillna('normal',inplace = True)
# # data['color'] = data['descrption'].str.extract('(durable)', expand=False)
# # data['durable'].fillna('normal',inplace = True)
# # tdata['durable'] = tdata['descrption'].str.extract('(durable)', expand=False)
# # tdata['durable'].fillna('normal',inplace = True)
# # data['sex'] = data['name'].str.extract('(women|Women|men|Men)', expand=False)
# # data['sex'].fillna('unknown',inplace = True)
# # data.loc[(data['sex']=='Women') | (data['sex']=='women'),'sex']='women'
# # data.loc[(data['sex']=='Men') | (data['sex']=='women'),'sex']='women'
# # tdata['fashion'] = tdata['name'].str.extract('(Fashion|fashion)', expand=False)
# # tdata['fashion'].fillna('unfashion',inplace = True)
# # tdata.loc[(tdata['fashion']=='Fashion') | (tdata['fashion']=='fashion'),'fashion']='fashion'

# def review_to_words(raw_review):   
#     letters_only = re.sub("[^a-zA-Z0-9.]", " ", raw_review) 
#     words = letters_only.lower().split()                             
#     return( " ".join( words )) 
# alldata = data
# alldata = pd.concat([data, tdata], axis=0,ignore_index=True)

alldata = data
alldata = alldata.append(tdata)

from collections import Counter


lvl3F = alldata['lvl3'].tolist()
lvl3F = list(set(lvl3F))
all_feature = np.array([])
values_counts = Counter()

for items in lvl3F:
    lvl3sp = alldata.loc[alldata['lvl3'] == items]
    dfname = lvl3sp['name'].tolist()
    getword = []
    for i in range(len(dfname)):
        for w in dfname[i].strip().split():  
            getword.append(w)
        values_counts = Counter(getword)
    top_20 = values_counts.most_common(40)
    for item in top_20:
        all_feature = np.append(all_feature,item[0])
all_feature = np.unique(all_feature)

# X=data[['lvl1','lvl2','lvl3','quality','new','fashion','durable','style','price','type']]
# tX=tdata[['lvl1','lvl2','lvl3','quality','new','fashion','durable','style','price','type']]
X=data[['lvl1','lvl2','lvl3','d_quality','d_new','price','type']]
tX=tdata[['lvl1','lvl2','lvl3','d_quality','d_new','price','type']]
y=label[['score']]
X =pd.get_dummies(X)
tX = pd.get_dummies(tX)
X['name']=data['name']
tX['name']=tdata['name']

def func(x, word):
    for item in re.split(r'[\s]',x) :
        if item == word :
            return 1
    return 0
for word in all_feature:
    X[word] = X['name'].apply(lambda x : func(x, word))
    tX[word] = tX['name'].apply(lambda x : func(x, word))
X.drop(['name'],axis=1,inplace=True)
tX.drop(['name'],axis=1,inplace=True)

x_columns = [x for x in X.columns]
X = X[x_columns]
y = y['score']
tx_columns = [x for x in tX.columns]
tX = tX[tx_columns]

from sklearn.model_selection import train_test_split
from sklearn.datasets import  make_classification
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

model = lgb.train({
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'num_leaves': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'metric': 'binary_logloss',
    'max_depth':7,
    'max_bin':300
}, lgb_train, num_boost_round=5000, valid_sets=lgb_eval, early_stopping_rounds=100, verbose_eval=100)

y_pred = model.predict(tX, num_iteration=model.best_iteration)

Y2 = tdata['id'].values
rfr_submission = pd.DataFrame({'id': Y2, 'Score': y_pred})
rfr_submission.to_csv('submission.csv', index=False)

# from sklearn.feature_extraction import DictVectorizer
# dict_vec = DictVectorizer(sparse=False)

# X_train = dict_vec.fit_transform(X.to_dict(orient='record'))
# X_test = dict_vec.transform(tX.to_dict(orient='record'))

# #from sklearn.ensemble import GradientBoostingRegressor
# rfr = RandomForestRegressor(n_estimators= 200, max_depth=75, min_samples_split=65,
#                                  min_samples_leaf=3,max_features=110,oob_score=True, random_state=100)
# #rfr = GradientBoostingRegressor()
# rfr.fit(X_train, y)
# rfr_y_predict = rfr.predict(X_test)
# #score = log_loss(y_valid, pred)

# Y2 = tdata['id'].values
# rfr_submission = pd.DataFrame({'id': Y2, 'Score': rfr_y_predict})
# rfr_submission.to_csv('/Users/ningsong/Desktop/all/submission.csv', index=False)

