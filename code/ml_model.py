#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:33:04 2019

@author: tunan
"""
'''
这里所有的都是
'''
import os,json,pickle
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor,plot_importance
#from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor,ExtraTreesRegressor,AdaBoostRegressor,RandomForestRegressor

class ml_model(object):
    '''
    base_model(estimator):定义使用什么模型
    adj_params:需要调整的参数
    params_dict:
    '''
    def __init__(self,base_model,adj_params,params_dict,int_feature,name):
        self.feature = None
        self.label = None
        self.model = None
        self.base_model = base_model   #未拟合前的模型。
        
        self.adj_params = adj_params
        self.params_dict = params_dict    #
        self.log_path = "../log/{}_config.json".format(name)   #最优参数的存储地址。
        self.model_path = "../models/{}_model".format(name)    #模型地址
        self.name = name
        self.int_feature = int_feature  #model中整数参数。
        
        if not os.path.exists('../log'):
            os.makedirs('../log')
            os.makedirs('../models')
            
    def load_model_with_params(self):
        #读取文件中的最优参数，并返回设置好参数的模型
        if not os.path.exists(self.log_path):
            raise ValueError("no file in {}".format(self.log_path))
            return None
        self.result = json.load(open(self.log_path,'r'))
        model = self.base_model(**self.result['params'])
        return model
            
    def auc_evaluate(self,**kwarg):
        '''
        description：给定参数下，模型的得分。
        inputs：
            kwarg(dict):需要调整的参数与其对应值。
        '''
        #固定的参数,与需调整的参数更新。得到模型完整参数params_dict.
        params_dict = self.params_dict  
        params_dict.update(kwarg)
        
        #将参数中整数型参数转换为整数。
        for f in self.int_feature:
            params_dict[f] = int(params_dict[f])
        
        #5折交叉验证，返回该参数下的模型评分。
        kf = KFold(5,shuffle=True,random_state=42)
        score = cross_val_score(self.base_model(**params_dict),
                                self.feature,self.label,
                                scoring = 'neg_mean_squared_error',cv = kf)
        return score.mean()
    
    def select_params(self):
        '''
        description:贝叶斯优化得到最优的参数。与固定参数合并后，将整数型参数转换并储存下来。
        '''
        bayes = BayesianOptimization(self.auc_evaluate,self.adj_params)
        bayes.maximize()
        result = bayes.max
        
        #与固定参数合并
        best_params = self.params_dict
        best_params.update(result['params'])
        
        #转换其中的整数型参数。
        int_feature = self.int_feature
        for f in int_feature:
            best_params[f] = int(best_params[f])
        result['params'] = best_params
        
        #最优参数存到log路径中。
        self.result = result
        with open(self.log_path,'w') as f:
            f.write(json.dumps(result))
            
    def fit(self,X_train,y_train):
        ''' 
        description:
            1. 传入特征和标签，使用贝叶斯优化得到最优参数.
            2. 最有参数的模型拟合特征和标签训练最优模型，并存入文件。
            3. 清空特征和标签。
            4. 如果已有模型文件就直接读出。
        '''
        
        if not os.path.exists(self.model_path):
            self.feature = X_train
            self.label = y_train
            print('start training {} model!'.format(self.name))
            self.select_params()
            model = self.base_model(**self.result['params'])
            model.fit(X_train,y_train)
            pickle.dump(model,open(self.model_path,'wb'))
            self.feature = None
            self.label = None
        else:
            print("loading {} model from file".format(self.name))
            model = pickle.load(open(self.model_path,'rb'))
            self.result = json.load(open(self.log_path,'r'))  #将最佳参数和结果也导出来
        
        self.model = model

    def predict(self,X_test):
        return self.model.predict(X_test)
    
    def dump_result(self,x):
        #将预测的结果导出
        if not os.path.exists("../result"):
            os.makedirs("../result")
       
        print("generate predicting result!")     
        y_update = self.predict(x)
        result = "\n".join([str(x) for x in y_update])

        path = "../result/{}_result.txt".format(self.name)
        with open(path,'w') as f:
            f.write(result)  
    
    def _print_analyse(self,x_test,y_test,save_img = False):
        y_pred = self.predict(x_test)
        mse = mean_squared_error(y_test,y_pred)
        print("{}交叉验证结果:{:.4f},验证集上结果:{:.4f}".format(\
              self.name,self.result['target'],mse))
        
        
class ensemble_model(object):
    def __init__(self,models,name):
        self.models = models
        self.name = name
    
    def predict(self,X_test):
        preds = []
        for model in self.models:
            pred = model.predict(X_test)
            preds.append(pred.reshape(-1,1))
        
        preds = np.concatenate(preds,axis = -1)
        return np.mean(preds,axis = -1)
            
    def _print_analyse(self,x_test,y_test):
        y_pred = self.predict(x_test)
        mse = mean_squared_error(y_test,y_pred)
        print("{}验证集上结果:{:.4f}".format(self.name,mse))
        
        
    def dump_result(self,x):
        #将预测的结果导出
        if not os.path.exists("../result"):
            os.makedirs("../result")
       
        print("generate predicting result!")     
        y_update = self.predict(x)
        result = "\n".join([str(x) for x in y_update])

        path = "../result/{}_result.txt".format(self.name)
        with open(path,'w') as f:
            f.write(result)  



if __name__ == '__main__':
    #获取数据，训练数据和验证数据。
    train_data = pd.read_csv("../inputs/zhengqi_train.txt",sep = '\t')
    x_update = pd.read_csv("../inputs/zhengqi_test.txt",sep = '\t').values
  
    X = train_data.values[:,:-1]
    y = train_data.values[:,-1]
    #打乱了
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("X_train shape is {}".format(X_train.shape))
    print("y_train shape is {}".format(y_train.shape))
    print("X_test shape is {}".format(X_test.shape))
    print("y_test shape is {}".format(y_test.shape))
        
    #lightgbm
    adj_dict = {'max_depth':(5,50),'n_estimators':(50,500),'learning_rate':(0.001,0.1),
                'num_leaves':(32,512),'min_child_samples':(20,100),'min_child_weight':(0.001,0.1),
                'feature_fraction':(0.5,1),'bagging_fraction':(0.5,1),'reg_alpha':(0,0.5),'reg_lambda':(0,0.5)}
    params_dict = {'objective':"regression" ,'max_bin':200,'verbose':1,'metric':['rmse']}
    int_feature = ['n_estimators','max_depth','num_leaves','min_child_samples']
    light_model = ml_model(LGBMRegressor,adj_dict,params_dict,int_feature = int_feature,name = 'lightgbm')
    light_model.fit(X_train,y_train)
    light_model._print_analyse(X_test,y_test)
    light_model.dump_result(x_update)
    
    #xgboost
    adj_dict = {'n_estimators':(50,500),'max_depth':(3,50),'subsample':(0.5,1),
                'reg_alpha':(0.1,1),'reg_lambda':(0.1,1),'learning_rate':(0.001,0.3)}
    params_dict = {'min_child_weight':1, 'seed':0,'colsample_bytree':0.8, 'gamma':0,'silent':1}
    int_feature = ['n_estimators','max_depth']
    xgb_model = ml_model(XGBRegressor,adj_dict,params_dict,int_feature = int_feature,name = 'xgboost')
    xgb_model.fit(X_train,y_train)
    xgb_model._print_analyse(X_test,y_test)
    xgb_model.dump_result(x_update)
    
    #gbdt
    adj_dict = {'max_depth':(3,50),'min_samples_split':(0.0001,0.01),
                'subsample':(0.5,1),'learning_rate':(0.0001,0.1),'n_estimators':(50,500)}
    params_dict = {'random_state':1,'max_features':'sqrt','verbose':0}
    int_feature = ['max_depth','n_estimators']
    gbdt_model = ml_model(GradientBoostingRegressor,adj_dict,params_dict,int_feature = int_feature,name = 'gbdt')
    gbdt_model.fit(X_train,y_train)
    gbdt_model._print_analyse(X_test,y_test)
    gbdt_model.dump_result(x_update)
        
    #random forest
    adj_dict = {"max_depth":(3,50),'n_estimators':(50,500)}
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
    
# =============================================================================
#     #SVM
#     adj_dict = {"C":(0.01,1000),'gamma':(0.001,1),'tol':(0.001,1.)}
#     params_dict = {'shrinking':True, 'kernel':'rbf','max_iter':-1}
#     svm_model = ml_model(SVR,adj_dict,params_dict,int_feature=[],name = 'svm')
#     svm_model.fit(X_train,y_train)
#     svm_model._print_analyse(X_test,y_test)
#     svm_model.dump_result(x_update)
# 
#     #adaboost
#     adj_dict = {"learning_rate":(0.001,0.3),'n_estimators':(50,500)}
#     params_dict = {"random_state":1}
#     int_feature = ["n_estimators"]
#     ada_model = ml_model(AdaBoostRegressor,adj_dict,params_dict,int_feature=int_feature,name = 'adaboost')
#     ada_model.fit(X_train,y_train)
#     ada_model._print_analyse(X_test,y_test)
#     ada_model.dump_result(x_update)
# =============================================================================
    

# =============================================================================
#     plot_importance(xgb_model.model)
#     from matplotlib import pyplot
#     pyplot.show()
#     importance = xgb_model.model.feature_importances_
#     print(pd.DataFrame({
#             'column': train_data.columns[:-1],
#             'importance': importance,
#         }).sort_values(by='importance'))
# =============================================================================
    
