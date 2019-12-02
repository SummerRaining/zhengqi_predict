#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:53:48 2019

@author: tunan
"""
import warnings,os,json
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

from ml_model import ml_model,ensemble_model
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor,ExtraTreesRegressor,AdaBoostRegressor,RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor 

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from util import print_analyse


def dump_result(y_update,name):
    #将预测的结果导出
    if not os.path.exists("../result"):
        os.makedirs("../result")
        
    result = "\n".join([str(x) for x in y_update])
    path = "../result/{}_result.txt".format(name)
    with open(path,'w') as f:
        f.write(result)  


# function to get training samples
def get_training_data():
    # extract training samples
    df_train = data_all[data_all["oringin"]=="train"]
    # split SalePrice and features
    y = df_train.target.values
    X = df_train.drop(["oringin","target"],axis=1)
    X = X.values
    X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train,X_valid,y_train,y_valid

# extract test data (without SalePrice)
def get_test_data():
    df_test = data_all[data_all["oringin"]=="test"].reset_index(drop=True)
    return df_test.drop(["oringin","target"],axis=1).values

def analyse_data_with_model(data,estimator,model_name,new_name):
    #模型参数从本地导入，最优化后的模型参数model_name.
    #训练后得到模型性能评估。
    X_train, X_test, y_train, y_test,x_update = data
    log_params = json.load(open("../log/{}_config.json".format(model_name),'r'))
    model = estimator(**log_params['params'])
    #5折交叉验证，返回该参数下的模型评分。
    kf = KFold(5,shuffle=True,random_state=42)
    score = cross_val_score(model,X_train,y_train,
                    scoring = 'neg_mean_squared_error',cv = kf)
    print("{}5折交叉验证的结果为{:.4f}".format(new_name,score.mean()))
    model.fit(X_train,y_train)
    print_analyse(y_test,model.predict(X_test),new_name)
    dump_result(model.predict(x_update),new_name)    
    

if __name__ == "__main__":
    #load_dataset
    with open("../inputs/zhengqi_train.txt")  as fr:
        data_train=pd.read_table(fr,sep="\t")
    with open("../inputs/zhengqi_test.txt") as fr_test:
        data_test=pd.read_table(fr_test,sep="\t")
        
    #merge train_set and test_set,用于区别是训练集还是测试集的数据。
    data_train["oringin"]="train"
    data_test["oringin"]="test"
    data_all=pd.concat([data_train,data_test],axis=0,ignore_index=True)

    data_all.drop(["V5","V9","V11","V17","V22","V28"],axis=1,inplace=True)
    X_train, X_test, y_train, y_test = get_training_data()
    x_update = get_test_data()
# =============================================================================
#     # Explore feature distibution 
#     #sns中的kedplot函数，用于显示数据的分布，并使用核估计拟合。三个参数，数据，颜色，阴影shade，返回图像轴。两个图像画在一起，第二个图像需要设置第一个的图像轴。
#     #通过图像轴设置x，y标签和图例
#     for column in data_all.columns[0:-2]:
#         g = sns.kdeplot(data_all[column][(data_all["oringin"] == "train")], color="Red", shade = True)
#         g = sns.kdeplot(data_all[column][(data_all["oringin"] == "test")], ax =g, color="Blue", shade= True)
#         g.set_xlabel(column)
#         g.set_ylabel("Frequency")
#         g = g.legend(["train","test"])
#         plt.show()
#         
#     #删除特征"V5","V9","V11","V17","V22","V28"，训练集和测试集分布不均
#     for column in ["V5","V9","V11","V17","V22","V28"]:
#         g = sns.kdeplot(data_all[column][(data_all["oringin"] == "train")], color="Red", shade = True)
#         g = sns.kdeplot(data_all[column][(data_all["oringin"] == "test")], ax =g, color="Blue", shade= True)
#         g.set_xlabel(column)
#         g.set_ylabel("Frequency")
#         g = g.legend(["train","test"])
#         plt.show()
# =============================================================================
    data = [X_train, X_test, y_train, y_test,x_update]
# =============================================================================
#     analyse_data_with_model(data,ExtraTreesRegressor,model_name = "extra_tree",new_name = "seleted_extra_tree")
#     analyse_data_with_model(data,RandomForestRegressor,model_name = "rf",new_name = "seleted_rf")
#     analyse_data_with_model(data,GradientBoostingRegressor,model_name = "gbdt",new_name = "seleted_gbdt")
#     analyse_data_with_model(data,XGBRegressor,model_name = "xgboost",new_name = "seleted_xgboost")
#     analyse_data_with_model(data,LGBMRegressor,model_name = "lightgbm",new_name = "seleted_lightgbm")
# =============================================================================
    
    #每次重新训练超参数,直接对原类进行修改
    #lightgbm
    adj_dict = {'max_depth':(1,8),'n_estimators':(100,800),'learning_rate':(0.001,0.1),
                'num_leaves':(32,512),'min_child_samples':(20,100),'min_child_weight':(0.001,0.1),
                'feature_fraction':(0.5,1),'bagging_fraction':(0.5,1),'reg_alpha':(0,0.5),'reg_lambda':(0,0.5)}
    params_dict = {'objective':"regression" ,'max_bin':200,'verbose':1,'metric':['rmse']}
    int_feature = ['n_estimators','max_depth','num_leaves','min_child_samples']
    light_model = ml_model(LGBMRegressor,adj_dict,params_dict,int_feature = int_feature,name = 'lightgbm')
    light_model.fit(X_train,y_train)
    light_model._print_analyse(X_test,y_test)
    light_model.dump_result(x_update)
    
    #xgboost
    adj_dict = {'n_estimators':(100,800),'max_depth':(1,10),'subsample':(0.5,1),
                'reg_alpha':(0.1,1),'reg_lambda':(0.1,1),'learning_rate':(0.001,0.3)}
    params_dict = {'min_child_weight':1, 'seed':0,'colsample_bytree':0.8, 'gamma':0,'silent':1}
    int_feature = ['n_estimators','max_depth']
    xgb_model = ml_model(XGBRegressor,adj_dict,params_dict,int_feature = int_feature,name = 'xgboost')
    xgb_model.fit(X_train,y_train)
    xgb_model._print_analyse(X_test,y_test)
    xgb_model.dump_result(x_update)
    
    #gbdt
    adj_dict = {'max_depth':(1,8),'min_samples_split':(0.0001,0.01),
                'subsample':(0.5,1),'learning_rate':(0.0001,0.1),'n_estimators':(100,800)}
    params_dict = {'random_state':1,'max_features':'sqrt','verbose':0}
    int_feature = ['max_depth','n_estimators']
    gbdt_model = ml_model(GradientBoostingRegressor,adj_dict,params_dict,int_feature = int_feature,name = 'gbdt')
    gbdt_model.fit(X_train,y_train)
    gbdt_model._print_analyse(X_test,y_test)
    gbdt_model.dump_result(x_update)
        
    #random forest
    adj_dict = {"max_depth":(1,15),'n_estimators':(50,500)}
    params_dict = {'verbose':0}
    int_feature = ['max_depth','n_estimators']
    rf_model = ml_model(RandomForestRegressor,adj_dict,params_dict,int_feature = int_feature,name = 'rf')
    rf_model.fit(X_train,y_train)
    rf_model._print_analyse(X_test,y_test)
    rf_model.dump_result(x_update)
    
    #extra_tree
    adj_dict = {'max_depth':(5,50),'max_features':(0.5,1.0),'min_samples_leaf':(5,30),
                'min_samples_split':(10,70) ,'n_estimators':(50,500)}
    params_dict = {'max_leaf_nodes':None,'min_impurity_decrease':0.0,
                   'min_weight_fraction_leaf':0,'bootstrap':False}
    int_feature = ['n_estimators','max_depth','min_samples_leaf','min_samples_split']
    et_model = ml_model(ExtraTreesRegressor,adj_dict,params_dict,int_feature = int_feature,name = 'extra_tree')
    et_model.fit(X_train,y_train)
    et_model._print_analyse(X_test,y_test)
    et_model.dump_result(x_update)
        
    #集成
    e_model = ensemble_model(models = [light_model,xgb_model,gbdt_model,rf_model,xgb_model],
                             name = 'model_ensemble')
    e_model._print_analyse(X_test,y_test)
    e_model.dump_result(x_update)
