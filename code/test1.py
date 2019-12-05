#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 20:48:36 2019

@author: tunan
"""

if __name__ == "__main__":
    #实验1
    svr.fit(X_train_head, y_train) #拟合SVR
    y_pred = svr.predict(X_test_head)
    result = pd.DataFrame(y_pred)
    result.to_csv("../feature_result/result(svr).txt", index=False, header=False)
    
    KRR2.fit(X_train_head, y_train)
    y_pred = KRR2.predict(X_test_head)
    result = pd.DataFrame(y_pred)
    result.to_csv("../feature_result/result(KRR2).txt", index=False, header=False)
    
    lgbm.fit(X_train_head, y_train)
    y_pred = lgbm.predict(X_test_head)
    result = pd.DataFrame(y_pred)
    result.to_csv("../feature_result/result(lgbm).txt", index=False, header=False)
    
    nn.fit(X_train_head, y_train)
    y_pred = nn.predict(X_test_head)
    result = pd.DataFrame(y_pred)
    result.to_csv("../feature_result/result(nn).txt", index=False, header=False)
    
    #xgboost
    score = rmsle_cv(xgb,X_train_head, y_train)
    print("Xgboost 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    xgb.fit(X_train_head, y_train)
    y_pred = xgb.predict(X_test_head)
    result = pd.DataFrame(y_pred)
    result.to_csv("../feature_result/result(xgb).txt", index=False, header=False)
    
    
    #实验2,不使用单变量选择univariance select
    os.makedirs("../univariace_result")
    
    score = rmsle_cv(KRR2, X_train, y_train)
    print("Kernel Ridge2 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    KRR2.fit(X_train, y_train)
    y_pred = KRR2.predict(X_test)
    result = pd.DataFrame(y_pred)
    result.to_csv("../univariace_result/result(KRR2).txt", index=False, header=False)
    
    score = rmsle_cv(lgbm,X_train, y_train)
    print("LGBM 得分: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    result = pd.DataFrame(y_pred)
    result.to_csv("../univariace_result/result(lgbm).txt", index=False, header=False)
    
    score = rmsle_cv(svr, X_train, y_train) #5折交叉验证
    print("SVR 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std())) #打印得分
    svr.fit(X_train, y_train) #拟合SVR
    y_pred = svr.predict(X_test)
    result = pd.DataFrame(y_pred)
    result.to_csv("../univariace_result/result(svr).txt", index=False, header=False)
    
    keras.backend.clear_session()
    score = rmsle_cv(nn,X_train, y_train)
    print("NN 得分: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    result = pd.DataFrame(y_pred)
    result.to_csv("../univariace_result/result(nn).txt", index=False, header=False)
    
    #xgboost
    score = rmsle_cv(xgb,X_train, y_train)
    print("Xgboost 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    result = pd.DataFrame(y_pred)
    result.to_csv("../univariace_result/result(xgb).txt", index=False, header=False)
    
    #实验3,不使用方差选择variance select
    os.makedirs("../variance_result")
    
    score = rmsle_cv(svr, X_train_head, y_train) #5折交叉验证
    print("SVR 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std())) #打印得分
    svr.fit(X_train_head, y_train) #拟合SVR
    y_pred = svr.predict(X_test_head)
    result = pd.DataFrame(y_pred)
    result.to_csv("../variance_result/result(svr).txt", index=False, header=False)
    
    score = rmsle_cv(KRR2, X_train_head, y_train)
    print("Kernel Ridge2 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    KRR2.fit(X_train_head, y_train)
    y_pred = KRR2.predict(X_test_head)
    result = pd.DataFrame(y_pred)
    result.to_csv("../variance_result/result(KRR2).txt", index=False, header=False)
    
    score = rmsle_cv(lgbm,X_train_head, y_train)
    print("LGBM 得分: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
    lgbm.fit(X_train_head, y_train)
    y_pred = lgbm.predict(X_test_head)
    result = pd.DataFrame(y_pred)
    result.to_csv("../variance_result/result(lgbm).txt", index=False, header=False)
    
    score = rmsle_cv(nn,X_train_head, y_train)
    print("NN 得分: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
    nn.fit(X_train_head, y_train)
    y_pred = nn.predict(X_test_head)
    result = pd.DataFrame(y_pred)
    result.to_csv("../variance_result/result(nn).txt", index=False, header=False)
    
    #xgboost
    score = rmsle_cv(xgb,X_train_head, y_train)
    print("Xgboost 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    xgb.fit(X_train_head, y_train)
    y_pred = xgb.predict(X_test_head)
    result = pd.DataFrame(y_pred)
    result.to_csv("../variance_result/result(xgb).txt", index=False, header=False)
    
# =============================================================================
#     #不做方差选择得到的特征数: 
#     ['V0', 'V1', 'V2', 'V3', 'V4', 'V6', 'V7', 'V8', 'V10', 'V12', 'V16',
#          'V20', 'V23', 'V24', 'V27', 'V31', 'V36', 'V37']
#     #做方差选择后得到的特征数。
#     ['V0', 'V1', 'V2', 'V3', 'V4', 'V6', 'V7', 'V8', 'V10', 'V12', 'V13',
#        'V16', 'V20', 'V23', 'V24', 'V31', 'V36', 'V37'],
# =============================================================================
    
    #实验3.2,使用全变量的树形模型
    os.makedirs("../all_feature_result")
    
    score = rmsle_cv(lgbm,X_train, y_train)
    print("LGBM 得分: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    result = pd.DataFrame(y_pred)
    result.to_csv("../all_feature_result/result(lgbm).txt", index=False, header=False)
    
    #xgboost
    score = rmsle_cv(xgb,X_train, y_train)
    print("Xgboost 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    result = pd.DataFrame(y_pred)
    result.to_csv("../all_feature_result/result(xgb).txt", index=False, header=False)
    
    #实验四,不使用纠正偏态分布后的结果
    os.makedirs("../skweness_result")
    models_ = [KRR2,lgbm,svr,nn,xgb]
    names_ = ['krr2','lgbm','svr','nn','xgboost']
    for model,name in zip(models_,names_):
        score = rmsle_cv(model, X_train_head, y_train)
        print("{} 得分: {:.4f} ({:.4f})\n".format(name,score.mean(), score.std()))
        model.fit(X_train_head, y_train)
        y_pred = model.predict(X_test_head)
        result = pd.DataFrame(y_pred)
        result.to_csv("../skweness_result/result({}).txt".format(name), index=False, header=False)
        
    

    