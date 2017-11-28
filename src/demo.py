#coding:utf8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#import tqdm
import datetime
import lightgbm as lgb
from sklearn.metrics import recall_score

validation=True
get_val=False
sub_sample=True
cross_feature=False
#unix_ts = 1439111214.0
#time = datetime.datetime.fromtimestamp(unix_ts)
negative_number=230

#读取数据----------------------------------------------------
#8477个shop
#97个mall_id，每个mall包含70个shop_id左右

# 获取wifi最强的ssd与强度
def get_wifi(wifi_infos):
    wifi_info_vector=wifi_infos.split(";")
    max_wifi=""
    max_value=-100000
    connect_wifi=""

    for wifi_info in wifi_info_vector:
        strs=wifi_info.split("|")
        wifi_name,wifi_strength,wifi_connected=strs
        if(wifi_connected=='true'):
            connect_wifi=wifi_name
        wifi_strength=int(wifi_strength)
        if(wifi_strength>max_value):
            max_wifi=wifi_name
            max_value=wifi_strength

    return connect_wifi+str("|")+max_wifi




shop_df=pd.read_csv("../data/训练数据-ccf_first_round_shop_info.csv",sep=",")
# 获取一个mall的shop数目
mall_shop_num=pd.DataFrame(shop_df[["mall_id","shop_id"]].groupby("mall_id").count()).rename(columns={"shop_id":"mall_shop_num"})
shop_df=shop_df.merge(mall_shop_num,left_on="mall_id",right_index=True,how="left")

train_df=pd.read_csv("../data/训练数据-ccf_first_round_user_shop_behavior.csv",sep=",")


train_df=train_df.sort_values(by="time_stamp")
#获取用户当前连接的wifi信息，以及信号最强的wifi信息
wifi_series=train_df['wifi_infos'].apply(get_wifi)
train_df["connected_wifi"]=wifi_series.apply(lambda x:x.split("|")[0])
train_df["max_wifi"]=wifi_series.apply(lambda x:x.split("|")[1])
#将时间中的unix时间戳转化为：周，时刻，周一-周五，周六，周日
train_df["weekday"]=train_df["time_stamp"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d %H:%M").weekday())
train_df["hour"]=train_df["time_stamp"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d %H:%M").hour)
#train_df["weekend"]=train_df["weekday"].apply(lambda x:)

temp_train_df = train_df.loc[(train_df["time_stamp"] < "2017-08-18 00:00")].copy()
temp_train_df=pd.merge(temp_train_df,shop_df,on="shop_id",how="left")

print ("-----------------------------------train prepropressed-----------------------------------")
#train_df["label"]=1
if validation==True:
    #按照时间顺序构建验证测试集合

    train_df.index=range(0,train_df.shape[0])

    #train_num=int(train_df.shape[0]*0.8)

    #test_df=train_df.ix[train_num:].copy()
    test_df=train_df.loc[train_df["time_stamp"]>="2017-08-25 00:00"]
    test_df['row_id']=test_df.index
    #train_df=train_df.ix[0:train_num].copy()
    train_df=train_df.loc[(train_df["time_stamp"]<"2017-08-25 00:00") & (train_df["time_stamp"]>="2017-08-18 00:00")]
    test_shop_id=test_df[["row_id","shop_id"]].copy()
    test_df=test_df.merge(shop_df[["mall_id","shop_id"]],how="left",on="shop_id")
    del test_df["shop_id"]

    #del test_df["shop_id_y"]

else:
    if sub_sample == True:
        train_df = train_df.loc[train_df["time_stamp"] >= "2017-08-18 00:00"]
        print (train_df.shape)

    test_df=pd.read_csv("../data/AB榜测试集-evaluation_public.csv",sep=",")

    #获取用户当前连接的wifi信息，以及信号最强的wifi信息
    wifi_series=test_df['wifi_infos'].apply(get_wifi)
    test_df["connected_wifi"]=wifi_series.apply(lambda x:x.split("|")[0])
    test_df["max_wifi"]=wifi_series.apply(lambda x:x.split("|")[1])
    #将时间中的unix时间戳转化为：周，时刻，周一-周五，周六，周日
    test_df["weekday"]=test_df["time_stamp"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d %H:%M").weekday())
    test_df["hour"]=test_df["time_stamp"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d %H:%M").hour)

    del test_df["wifi_infos"]
    del test_df["time_stamp"]
del train_df["wifi_infos"]
del train_df["time_stamp"]

print (train_df.columns)
print (test_df.columns)
print ("-----------------------------------test preprocessed-----------------------------------")
    #test_df.columns=Index([u'row_id', u'user_id', u'mall_id', u'longitude', u'latitude',u'connected_wifi', u'max_wifi', u'weekday', u'hour'],dtype='object')

mall_buy_num=pd.DataFrame(temp_train_df[["mall_id","shop_id"]].groupby("mall_id",as_index=False).count()).rename(columns={"shop_id":"mall_buy_num"})
shop_df=shop_df.merge(mall_buy_num,on="mall_id",how="left")

shop_buy_num=pd.DataFrame(temp_train_df[["mall_id","shop_id"]].groupby("shop_id",as_index=False).count()).rename(columns={"mall_id":"shop_buy_num"})
shop_df=shop_df.merge(shop_buy_num,on="shop_id",how="left")

category_id_buy_num=pd.DataFrame(temp_train_df[["category_id","shop_id"]].groupby("category_id",as_index=False).count()).rename(columns={"shop_id":"category_id_buy_num"})
shop_df=shop_df.merge(category_id_buy_num,on="category_id",how="left")
shop_df.fillna(0,inplace=True)

connected_wifi_buy_num=pd.DataFrame(temp_train_df[["connected_wifi","shop_id"]].groupby("connected_wifi",as_index=False).count()).rename(columns={"shop_id":"connected_wifi_buy_num"})

train_df=train_df.merge(connected_wifi_buy_num,on="connected_wifi",how="left")
test_df=test_df.merge(connected_wifi_buy_num,on="connected_wifi",how="left")
train_df.fillna(0,inplace=True)

max_wifi_buy_num=pd.DataFrame(temp_train_df[["max_wifi","shop_id"]].groupby("max_wifi",as_index=False).count()).rename(columns={"shop_id":"max_wifi_buy_num"})

train_df=train_df.merge(max_wifi_buy_num,on="max_wifi",how="left")
test_df=test_df.merge(max_wifi_buy_num,on="max_wifi",how="left")
test_df.fillna(0,inplace=True)

del temp_train_df
#构建 shop_id 到mall_id的相互映射关系
'''
shop_array=shop_df[['shop_id','mall_id']].values
dict_shop2mall={}
dict_mall2shaopset={}
for i in range(shop_array.shape[0]):
    [t_shop,t_mall]=shop_array[i,:]
    dict_shop2mall[t_shop]=t_mall
    if(dict_mall2shaopset.get(t_mall)==None):
        dict_mall2shaopset[t_mall]={}
        dict_mall2shaopset[t_mall][t_shop]=0
    else:
        dict_mall2shaopset[t_mall][t_shop] = 0
'''

#训练数据：基于正样本，构建负样本，并加入到样本集合里面
#对于没有给正样本，从他所在的shop_id选取距离最近的n个商店以及，随机选取m个其他商店
#1：生成一个dataframe---nearest_shop_df, 包含两列，shop_id，相邻的shop_id--n_shop_id
#①：shop_id到与其所在同一个mall的shop_id的映射
shop2shop_df=pd.merge(shop_df[["mall_id","shop_id","longitude","latitude"]],shop_df,on="mall_id")
shop2shop_df["distance1"]=shop2shop_df["longitude_x"]-shop2shop_df["longitude_y"]
shop2shop_df["distance2"]=shop2shop_df["latitude_x"]-shop2shop_df["latitude_y"]
shop2shop_df["distance"]=shop2shop_df["distance1"]*shop2shop_df["distance1"]+shop2shop_df["distance2"]*shop2shop_df["distance2"]
del shop2shop_df["distance1"]
del shop2shop_df["distance2"]
del shop2shop_df["longitude_x"]
del shop2shop_df["latitude_x"]
shop2shop_df.rename(columns={"latitude_y":"latitude","longitude_y":"longitude","shop_id_x":"shop_id"},inplace=True)
#shop2shop_df["rn"]=shop2shop_df[["shop_id","distance"]].groupby("shop_id").rank(method="average")  #生成排序号，
shop2shop_df["rn"]=shop2shop_df.sort_values(['distance'], ascending=[True]).groupby("shop_id").cumcount() + 1
nearest_shop_df=shop2shop_df.loc[(shop2shop_df["rn"]<=negative_number) & (shop2shop_df["rn"]!=1)].copy()
del nearest_shop_df["rn"]
del nearest_shop_df["distance"]
del shop2shop_df["rn"]
del shop2shop_df["distance"]


#3：将train与nearest_shop_df基于shop_id进行merge-生成负样本df
train_negative=pd.merge(train_df,nearest_shop_df,on="shop_id",how="left")
del train_negative["shop_id"]
train_negative.rename(columns={"shop_id_y":"shop_id"},inplace=True)
train_negative["label"]=0
train_positive=pd.merge(train_df,shop_df,on="shop_id",how="left")
train_positive["label"]=1
#获得完整的数据--包含正样本和负样本
train_all=pd.concat([train_negative,train_positive])
del  train_negative
del  train_positive

print ("----------------------------------get train_all-----------------------------------")

#获取两种shop_id之间的距离
train_all["distance1"]=train_all["longitude_x"]-train_all["longitude_y"]
train_all["distance2"]=train_all["latitude_x"]-train_all["latitude_y"]
train_all["distance"]=train_all["distance1"]*train_all["distance1"]+train_all["distance2"]*train_all["distance2"]

del train_all["distance1"]
del train_all["distance2"]

'''
del train_all["longitude_y"]
del train_all["latitude_y"]
del train_all["longitude_x"]
del train_all["latitude_x"]
'''

#test_data


#先尝试删除经纬度，保留距离，然后使用lightgbm训练，预测
cols1=["connected_wifi","max_wifi","weekday","hour"]

for col in cols1:
    #if train[col].dtype == 'object':
    print(col)
    train_all[col] = train_all[col].apply(str)
    test_df[col] = test_df[col].apply(str)

    le = LabelEncoder()
    train_vals = list(train_all[col].unique())
    test_vals = list(test_df[col].unique())
    le.fit(train_vals + test_vals)
    train_all[col] = le.transform(train_all[col])
    test_df[col] = le.transform(test_df[col])

#获取训练数据，进行训练
cols2=["category_id","mall_id","shop_id"]

for col in cols2:
    #if train[col].dtype == 'object':
    print(col)
    value_list=list(shop_df[col].unique())
    le = LabelEncoder()
    le.fit(value_list)

    train_all[col] = le.transform(train_all[col])
    #test_df[col] = le.transform(test_df[col])

if cross_feature==True:
    print ("-----------------------------------get cross_feature-----------------------------------")
    train_all["max_wifi_shop_id"]=train_all["max_wifi"]*10000+train_all["shop_id"]
    train_all["connected_wifi_shop_id"]=train_all["connected_wifi"]*10000+train_all["shop_id"]


train_cols=list(train_all.columns)
train_cols.remove("label")
train_cols.remove("user_id")
X=train_all[train_cols].values
y=train_all["label"].values
feature_name=train_cols
categorical_feature=train_cols[:]
categorical_feature.remove("price")
categorical_feature.remove("distance")
categorical_feature.remove("mall_shop_num")
#categorical_feature.remove("distance1")
#categorical_feature.remove("distance2")
categorical_feature.remove("longitude_x")
categorical_feature.remove("latitude_x")
categorical_feature.remove("longitude_y")
categorical_feature.remove("latitude_y")
categorical_feature.remove("mall_buy_num")
categorical_feature.remove("shop_buy_num")
categorical_feature.remove("category_id_buy_num")
categorical_feature.remove("connected_wifi_buy_num")
categorical_feature.remove("max_wifi_buy_num")


print (feature_name)
print (categorical_feature)

del train_all

print ("-----------------------------------get val data-----------------------------------")
if get_val==True:
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.8, random_state=12)
    d_train = lgb.Dataset(X_train, label=y_train,feature_name=feature_name,categorical_feature=categorical_feature)
    d_valid = lgb.Dataset(X_valid, label=y_valid,feature_name=feature_name,categorical_feature=categorical_feature)
else:
    d_train = lgb.Dataset(X, label=y,feature_name=feature_name,categorical_feature=categorical_feature)
    del X
#d_valid = lgb.Dataset(X_valid, label=y_valid,feature_name=feature_name,categorical_feature=categorical_feature)
#watchlist = [d_train, d_valid]

print('Training LGBM model...')
params = {}
params['learning_rate'] = 0.1
params['application'] = 'binary'
params['max_depth'] = 14
params['num_leaves'] = 2 ** 12
params['verbosity'] = 1
#params['metric'] = 'auc'
#params['watchlist']=[d_train, d_valid]
model = lgb.train(params, train_set=d_train, num_boost_round=200,feature_name=feature_name,
                  categorical_feature=categorical_feature)
if get_val==True:
    print('Making validation and saving them...')
    p_test = model.predict(X_valid)
    pY=np.zeros(p_test.shape[0])
    pY[p_test>0.5]=1
    print((pY==y_valid).mean())
    print (recall_score(y_true=y_valid,y_pred=pY))
    print("p_test.shape",p_test.shape)

#----------------------------------------测试，并生成测试结果----------------------------------
#将测试集分成10份，分十次预测
del d_train
print ("-----------------------------------test-----------------------------------")

# cut_number=10
# test_df.index=range(0,test_df.shape[0])
# part_num=test_df.shape[0]/cut_number
# i=0
# result=pd.DataFrame(columns=["row_id","shop_id"])
# while i<cut_number:
#
#     print (i)
#     if i==(cut_number-1):
#         part_test_df = test_df.iloc[i * part_num:]
#     else:
#         part_test_df=test_df.iloc[i*part_num:part_num*(i+1)]
#     part_test_df_all=pd.merge(part_test_df,shop_df,on="mall_id",how="left")
#
#     part_test_df_all["distance1"]=part_test_df_all["longitude_x"]-part_test_df_all["longitude_y"]
#     part_test_df_all["distance2"]=part_test_df_all["latitude_x"]-part_test_df_all["latitude_y"]
#     part_test_df_all["distance"]=part_test_df_all["distance1"]*part_test_df_all["distance1"]+part_test_df_all["distance2"]*part_test_df_all["distance2"]
#
#     del part_test_df_all["distance1"]
#     del part_test_df_all["distance2"]
#
#
#     '''
#     del part_test_df_all["longitude_y"]
#     del part_test_df_all["latitude_y"]
#     del part_test_df_all["longitude_x"]
#     del part_test_df_all["latitude_x"]
#     '''
#     if cross_feature == True:
#         print ("-----------------------------------get cross_feature-----------------------------------")
#         part_test_df_all["max_wifi_shop_id"] = part_test_df_all["max_wifi"] * 10000 + train_all["shop_id"]
#         part_test_df_all["connected_wifi_shop_id"] = part_test_df_all["connected_wifi"] * 10000 + part_test_df_all["shop_id"]
#         print (part_test_df_all["max_wifi_shop_id"].dtype)

#     part_test_pre_df=part_test_df_all[["row_id","shop_id"]]
#
#     cols2=["category_id","mall_id","shop_id"]
#
#     for col in cols2:
#         #if train[col].dtype == 'object':
#         print(col)
#         value_list=list(shop_df[col].unique())
#         le = LabelEncoder()
#         le.fit(value_list)
#
#         part_test_df_all[col] = le.transform(part_test_df_all[col])
#         #test_df[col] = le.transform(test_df[col])
#
#     part_test=part_test_df_all[train_cols].values
#     p_test = model.predict(part_test)
#     pY=np.zeros(p_test.shape[0])
#     part_test_pre_df["pro"]=p_test
#     part_test_pre_df["rn"] = part_test_pre_df.sort_values(['pro'], ascending=[False]).groupby("row_id").cumcount() + 1
#     part_pro_shop=part_test_pre_df.loc[part_test_pre_df["rn"]==1,["row_id","shop_id"]].copy()
#     result=pd.concat([result,part_pro_shop])
#     i += 1
#
#     if validation == True:
#         val_result = pd.merge(test_shop_id, result, on="row_id")
#
#         val_result["acc"] = val_result["shop_id_x"] == val_result["shop_id_y"]
#         print (val_result["acc"].mean())
#
#
# if validation==True:
#     val_result=pd.merge(test_shop_id,result,on="row_id")
#
#     val_result["acc"]=val_result["shop_id_x"]==val_result["shop_id_y"]
#     print (val_result["acc"].mean())
# else:
#     result.to_csv("../result/result.csv",sep=",", index=False)

# 预测

# test的两个shop_经纬度之间的距离
test_all = pd.merge(test_df, shop_df, on="mall_id", how="left")
#test_df_all=test_df.merge(shop_df[["mall_id","shop_id"]],how="left",on="shop_id")
test_all["distance1"]=test_all["longitude_x"]-test_all["longitude_y"]
test_all["distance2"]=test_all["latitude_x"]-test_all["latitude_y"]
test_all["distance"]=test_all["distance1"]*test_all["distance1"]+test_all["distance2"]*test_all["distance2"]
del test_all["distance1"]
del test_all["distance2"]

test_pre_df=test_all[["row_id", "shop_id"]]

cols2=["category_id","mall_id","shop_id"]

for col in cols2:
    #if train[col].dtype == 'object':
    print(col)
    value_list=list(shop_df[col].unique())
    le = LabelEncoder()
    le.fit(value_list)

    test_all[col] = le.transform(test_all[col])


test = test_all[train_cols].values
p_test = model.predict(test)
pY = np.zeros(p_test.shape[0])
test_pre_df["pro"] = p_test
test_pre_df["rn"] = test_pre_df.sort_values(['pro'], ascending=[False]).groupby("row_id").cumcount() + 1
result = test_pre_df.loc[test_pre_df["rn"]==1, ["row_id", "shop_id"]].copy()

if validation:
    val_result = pd.merge(test_shop_id, result, on="row_id")
    val_result["acc"] = val_result["shop_id_x"] == val_result["shop_id_y"]
    print("valid acc:",val_result["acc"].mean())
else:
    result.to_csv("../result/result2.csv", sep=",", index=False)


