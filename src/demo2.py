#coding:utf8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#import tqdm
import datetime
import lightgbm as lgb
from sklearn.metrics import recall_score

negative_number = 230
cross_feature = False
valid = False

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

def get_wifi_v2(wifi_infos):
    wifi_info_vector=wifi_infos.split(";")
    wifi_strength_name = {}
    connect_wifi = ""

    for wifi_info in wifi_info_vector:
        strs=wifi_info.split("|")
        wifi_name,wifi_strength,wifi_connected = strs
        if(wifi_connected=='true'):
            connect_wifi=wifi_name
        wifi_strength=int(wifi_strength)
        wifi_strength_name[wifi_name] = wifi_strength

    max_wifi = []
    s = sorted(wifi_strength_name.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    if len(s) >= 4:
        for i in range(4):
            max_wifi.append(s[i][0])
    else:
        for i in range(len(s)):
            max_wifi.append(s[i][0])
        for i in range(len(s), 4):
            max_wifi.append("UK")

    return connect_wifi+str("|")+max_wifi[0]+str("|")+max_wifi[1]+str("|")+max_wifi[2]+str("|")+max_wifi[3]

# 读取数据
shop_df = pd.read_csv("../data/训练数据-ccf_first_round_shop_info.csv", sep=",")
train_df = pd.read_csv("../data/训练数据-ccf_first_round_user_shop_behavior.csv", sep=",")
test_df = pd.read_csv("../data/AB榜测试集-evaluation_public.csv", sep=",")
print("original train num:", len(train_df))
print("original test num:", len(test_df))
print("original shop num:", len(shop_df))
print("data load finish!")

test_shop_id = None # 存储验证集的label
#test_pre_df = None # 存储预测中间量
# 划分验证集测试集
if valid:
    train_df = train_df.sort_values(by="time_stamp")
    train_df.index = range(0, train_df.shape[0])
    test_df = train_df.loc[(train_df["time_stamp"] >= "2017-08-25 00:00")].copy()
    # 验证集是从train中划分而来，没有row_id
    test_df["row_id"] = test_df.index
    train_df = train_df.loc[(train_df["time_stamp"] < "2017-08-25 00:00")
                            #&(train_df["time_stamp"] > "2017-08-17 00:00")
                            ].copy()
    # 验证集是从train中划分而来，没有mall_id
    test_df = pd.merge(test_df, shop_df[["shop_id", "mall_id"]], on="shop_id", how="left")
    test_shop_id = test_df[["row_id", "shop_id"]].copy() # 注意，此时shopid未编码
    del test_df["shop_id"]

# train与test用户特征抽取
# train
wifi_series=train_df['wifi_infos'].apply(get_wifi_v2)
train_df["connected_wifi"]=wifi_series.apply(lambda x:x.split("|")[0])
train_df["max_wifi1"]=wifi_series.apply(lambda x:x.split("|")[1])
train_df["max_wifi2"]=wifi_series.apply(lambda x:x.split("|")[2])
train_df["max_wifi3"]=wifi_series.apply(lambda x:x.split("|")[3])
train_df["max_wifi4"]=wifi_series.apply(lambda x:x.split("|")[4])
train_df["weekday"]=train_df["time_stamp"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d %H:%M").weekday())
train_df["hour"]=train_df["time_stamp"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d %H:%M").hour)
#train_df["weekend"]=train_df["weekday"].apply(lambda x:)
del train_df["wifi_infos"]
del train_df["time_stamp"]

# test
wifi_series=test_df['wifi_infos'].apply(get_wifi_v2)
test_df["connected_wifi"]=wifi_series.apply(lambda x:x.split("|")[0])
test_df["max_wifi1"]=wifi_series.apply(lambda x:x.split("|")[1])
test_df["max_wifi2"]=wifi_series.apply(lambda x:x.split("|")[2])
test_df["max_wifi3"]=wifi_series.apply(lambda x:x.split("|")[3])
test_df["max_wifi4"]=wifi_series.apply(lambda x:x.split("|")[4])
test_df["weekday"]=test_df["time_stamp"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d %H:%M").weekday())
test_df["hour"]=test_df["time_stamp"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d %H:%M").hour)
del test_df["wifi_infos"]
del test_df["time_stamp"]

# 编码离散特征
cols=["connected_wifi", "max_wifi1", "max_wifi2", "max_wifi3", "max_wifi4", "weekday", "hour"]
for col in cols:
    #if train[col].dtype == 'object':
    print("encoding:", col)
    train_df[col] = train_df[col].apply(str)
    test_df[col] = test_df[col].apply(str)

    le = LabelEncoder()
    oneEnc = OneHotEncoder(categorical_features='all', sparse=True)

    train_vals = list(train_df[col].unique())
    test_vals = list(test_df[col].unique())

    le.fit(train_vals + test_vals)
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

    # oneEnc.fit(train_vals + test_vals)
    # train_df[col] = oneEnc.transform(train_df[col])
    # test_df[col] = oneEnc.transform(test_df[col])

print("encoding:", "category_id")
shop_df["category_id"] = shop_df["category_id"].apply(str)
le_catid = LabelEncoder()
cat_vals = list(shop_df["category_id"].unique())
le_catid.fit(cat_vals)
shop_df["category_id"] = le_catid.transform(shop_df["category_id"])

print("encoding:", "mall_id")
shop_df["mall_id"] = shop_df["mall_id"].apply(str)
le_mallid = LabelEncoder()
mall_vals = list(shop_df["mall_id"].unique())
le_mallid.fit(mall_vals)
shop_df["mall_id"] = le_mallid.transform(shop_df["mall_id"])
#train_df["mall_id"] = le_mallid.transform(train_df["mall_id"])
test_df["mall_id"] = le_mallid.transform(test_df["mall_id"])

print("encoding:", "shop_id")
shop_df["shop_id"] = shop_df["shop_id"].apply(str)
le_shopid = LabelEncoder()
shop_vals = list(shop_df["shop_id"].unique())
le_shopid.fit(shop_vals)
shop_df["shop_id"] = le_shopid.transform(shop_df["shop_id"])
train_df["shop_id"] = le_shopid.transform(train_df["shop_id"])
# if valid:
#     test_shop_id["shop_id"] = le_shopid.transform(test_shop_id["shop_id"])

# shop特征抽取
temp_train_df = pd.merge(train_df, shop_df, on="shop_id", how="left")

mall_shop_num = pd.DataFrame(shop_df[["mall_id", "shop_id"]].groupby("mall_id").count()).rename(columns={"shop_id":"mall_shop_num"})
shop_df = shop_df.merge(mall_shop_num, left_on="mall_id", right_index=True, how="left")
del mall_shop_num

mall_buy_num = pd.DataFrame(temp_train_df[["mall_id", "shop_id"]].groupby("mall_id",as_index=False).count()).rename(columns={"shop_id":"mall_buy_num"})
shop_df = shop_df.merge(mall_buy_num, on="mall_id", how="left")
del mall_buy_num

shop_buy_num = pd.DataFrame(temp_train_df[["mall_id", "shop_id"]].groupby("shop_id",as_index=False).count()).rename(columns={"mall_id":"shop_buy_num"})
shop_df = shop_df.merge(shop_buy_num, on="shop_id",how="left")
del shop_buy_num

category_id_buy_num = pd.DataFrame(temp_train_df[["category_id", "shop_id"]].groupby("category_id",as_index=False).count()).rename(columns={"shop_id":"category_id_buy_num"})
shop_df = shop_df.merge(category_id_buy_num, on="category_id",how="left")
del category_id_buy_num

# position_mean=temp_train_df[['shop_id', "longitude_x", "latitude_x"]].groupby("shop_id").mean().rename(columns={"longitude_x":"longitude_mean","latitude_x":"latitude_mean"})
# #temp_train_df = pd.merge(train_df, shop_df, on="shop_id", how="left")
# shop_df=shop_df.merge(position_mean, left_on="shop_id", right_index=True, how="left")
# shop_df['longitude_mean']=shop_df['longitude_mean'].fillna(value=shop_df['longitude'])
# shop_df['latitude_mean']=shop_df['latitude_mean'].fillna(value=shop_df['latitude'])

shop_df.fillna(0, inplace=True)
print("shop feature:", shop_df.columns)

# 用户统计特征抽取
connected_wifi_buy_num=pd.DataFrame(temp_train_df[["connected_wifi","shop_id"]].groupby("connected_wifi",as_index=False).count()).rename(columns={"shop_id":"connected_wifi_buy_num"})
train_df=train_df.merge(connected_wifi_buy_num,on="connected_wifi",how="left")
test_df=test_df.merge(connected_wifi_buy_num,on="connected_wifi",how="left")

max_wifi_buy_num=pd.DataFrame(temp_train_df[["max_wifi1","shop_id"]].groupby("max_wifi1",as_index=False).count()).rename(columns={"shop_id":"max_wifi_buy_num"})
train_df=train_df.merge(max_wifi_buy_num,on="max_wifi1",how="left")
test_df=test_df.merge(max_wifi_buy_num,on="max_wifi1",how="left")

test_df.fillna(0,inplace=True)
train_df.fillna(0,inplace=True)
del temp_train_df

print("train user feature:", train_df.columns)
print("test user feature:", test_df.columns)
print("train num:", len(train_df))
print("test num:", len(test_df))

# 近邻构建负样本
shop2shop_df = pd.merge(shop_df[["mall_id", "shop_id", "longitude", "latitude"]], shop_df, on="mall_id")
print("shop2shop columns:", shop2shop_df.columns)
shop2shop_df["distance1"] = shop2shop_df["longitude_x"] - shop2shop_df["longitude_y"]
shop2shop_df["distance2"] = shop2shop_df["latitude_x"] - shop2shop_df["latitude_y"]
shop2shop_df["distance"] = shop2shop_df["distance1"]*shop2shop_df["distance1"]+shop2shop_df["distance2"]*shop2shop_df["distance2"]
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
print("nearest_shop columns:", nearest_shop_df.columns)


#3：将train与nearest_shop_df基于shop_id进行merge-生成负样本df
train_negative=pd.merge(train_df, nearest_shop_df, on="shop_id",how="left")
del train_negative["shop_id"]
train_negative.rename(columns={"shop_id_y":"shop_id"}, inplace=True)
train_negative["label"]=0
train_positive=pd.merge(train_df, shop_df, on="shop_id", how="left")
train_positive["label"]=1
#获得完整的数据--包含正样本和负样本
train_all=pd.concat([train_negative,train_positive])
del  train_negative
del  train_positive

#train的两个shop_经纬度之间的距离
train_all["distance1"]=train_all["longitude_x"]-train_all["longitude_y"]
train_all["distance2"]=train_all["latitude_x"]-train_all["latitude_y"]
train_all["distance"]=train_all["distance1"]*train_all["distance1"]+train_all["distance2"]*train_all["distance2"]
del train_all["distance1"]
del train_all["distance2"]

train_all.index=range(train_all.shape[0])
train_all["row_id"] = train_all.index
temp=train_all[['row_id', 'distance']].sort_values(['distance'], ascending=[False]).groupby("row_id").cumcount() + 1
train_all["rn"] =temp.sort_index()
del temp


print("train_all feature:", train_all.columns)
print("train_all num:", len(train_all))

# test的两个shop_经纬度之间的距离
test_all = pd.merge(test_df, shop_df, on="mall_id", how="left")
#test_df_all=test_df.merge(shop_df[["mall_id","shop_id"]],how="left",on="shop_id")
test_all["distance1"]=test_all["longitude_x"]-test_all["longitude_y"]
test_all["distance2"]=test_all["latitude_x"]-test_all["latitude_y"]
test_all["distance"]=test_all["distance1"]*test_all["distance1"]+test_all["distance2"]*test_all["distance2"]
del test_all["distance1"]
del test_all["distance2"]

test_all.index=range(test_all.shape[0])
temp=test_all[['row_id', 'distance']].sort_values(['distance'], ascending=[False]).groupby("row_id").cumcount() + 1
test_all["rn"] =temp.sort_index()

del temp

print("current valid mode:", valid)
print("test_all features:", test_all.columns)
print("test_all num:", len(test_all))



# 构建lgb格式训练集
train_cols=list(train_all.columns)
train_cols.remove("label")
train_cols.remove("user_id")
train_cols.remove("row_id")
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

# 设置lgb参数并训练
print('Training LGBM model...')
params = {}
params['learning_rate'] = 0.05
params['application'] = 'binary'
params['max_depth'] = 30
params['num_leaves'] = 2 ** 14
params['verbosity'] = 1
#params['metric'] = 'auc'
#params['watchlist']=[d_train, d_valid]
d_train = lgb.Dataset(X, label=y,feature_name=feature_name,categorical_feature=categorical_feature)
model = lgb.train(params, train_set=d_train, num_boost_round=300,feature_name=feature_name,
                  categorical_feature=categorical_feature)
del d_train

# 预测
test_pre_df = test_all[["row_id", "shop_id"]]
test = test_all[train_cols].values
p_test = model.predict(test)
pY = np.zeros(p_test.shape[0])
test_pre_df["pro"] = p_test
test_pre_df["rn"] = test_pre_df.sort_values(['pro'], ascending=[False]).groupby("row_id").cumcount() + 1
result = test_pre_df.loc[test_pre_df["rn"]==1, ["row_id", "shop_id"]].copy()
result["shop_id"] = le_shopid.inverse_transform(result["shop_id"])

if valid:
    val_result = pd.merge(test_shop_id, result, on="row_id")
    val_result["acc"] = val_result["shop_id_x"] == val_result["shop_id_y"]
    print("valid acc:",val_result["acc"].mean())
else:
    result.to_csv("../result/result3.csv", sep=",", index=False)





