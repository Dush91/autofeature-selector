
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
        




def preprocess_dataset(dataset_path):
    # Your code starts here (Multiple lines)
    player_df = pd.read_csv(dataset_path)
    numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
    catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']
    player_df = player_df[numcols+catcols]
    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
    features = traindf.columns
    traindf = traindf.dropna()
    traindf = pd.DataFrame(traindf,columns=features)
    y = traindf['Overall']>=87
    X = traindf.copy()
    del X['Overall']
    num_feats=30  
    return X, y, num_feats




def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)
    
    # preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)
    feature_name = list(X.columns)
   
    def cor_selector(X, y,num_feats):
        cor_list = []
        for i in X.columns.tolist():
            cor = np.corrcoef(X[i], y)[0, 1]
            cor_list.append(cor)
        
        cor_list = [0 if np.isnan(i) else i for i in cor_list]
        cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
        cor_support = [True if i in cor_feature else False for i in feature_name]
        return cor_support, cor_feature
    
    def chi_squared_selector(X, y, num_feats):
        chi2_selector = SelectKBest(chi2,k = num_feats)
        topfeatures = chi2_selector.fit(X, y)
        dfscores = pd.DataFrame(topfeatures.scores_)
        dfcolumns = pd.DataFrame(X.columns)
        featureScores = pd.concat([dfcolumns,dfscores], axis=1)   
        featureScores.columns = ['Specs', 'Score']
        d = featureScores.nlargest(num_feats,'Score')
        chi_feature = d['Specs'].tolist()
        chi_support = [True if i in chi_feature else False for i in feature_name]
  
        return chi_support, chi_feature
    
    def rfe_selector(X, y, num_feats):
   
        mms = MinMaxScaler()
        X_sc = mms.fit_transform(X)
        lr = LogisticRegression(solver='lbfgs')
        rfe = RFE(estimator=lr,n_features_to_select=num_feats,step=1,verbose=5)
        rfe_lr = rfe.fit(X_sc, y)
        rfe_support = rfe_lr.get_support()
        rfe_support, type(rfe_support)
        rfe_feature = X.loc[:, rfe_support].columns.tolist()
        return rfe_support, rfe_feature
    
    def embedded_log_reg_selector(X, y, num_feats):
                                  
        mms = MinMaxScaler()
        X_sc = mms.fit_transform(X)
        logreg = LogisticRegression(penalty='l1', solver='liblinear')
        embedded_lr_selector = SelectFromModel(LogisticRegression(penalty='l2', solver='liblinear', max_iter=50000), max_features=num_feats)
        embedded_lr_selector = embedded_lr_selector.fit(X_sc, y)
        embedded_lr_support = embedded_lr_selector.get_support()
        embedded_lr_feature = X.loc[:, embedded_lr_support].columns.tolist()
    
        return embedded_lr_support, embedded_lr_feature
    
    def embedded_rf_selector(X, y, num_feats):
    
        mms = MinMaxScaler()
        X_sc = mms.fit_transform(X)
        rf = RandomForestClassifier(n_estimators=100)
        embedded_rf_selector = SelectFromModel(rf,max_features=num_feats)
        embedded_rf_selector = embedded_rf_selector.fit(X_sc, y)
        embedded_rf_support = embedded_rf_selector.get_support()
        embedded_rf_feature = X.loc[:, embedded_rf_support].columns.tolist()
    
        return embedded_rf_support, embedded_rf_feature
    
    def embedded_lgbm_selector(X, y, num_feats):
    
        lgbmc = LGBMClassifier(n_estimators=500,
                      learning_rate=0.05,
                      num_leaves=32,
                      colsample_bytree=0.2,
                      reg_alpha=3,
                      reg_lambda=1,
                      min_split_gain=0.01,
                      min_child_weight=40)
    
        embedded_lgbm_selector = SelectFromModel(lgbmc,max_features=num_feats)
        embedded_lgbm_selector = embedded_lgbm_selector.fit(X, y)
        embedded_lgbm_support = embedded_lgbm_selector.get_support()
        embedded_lgbm_feature = X.loc[:, embedded_lgbm_support].columns.tolist()
    
        return embedded_lgbm_support, embedded_lgbm_feature
    
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
    
    
    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    #### Your Code starts here (Multiple lines)
    pd.set_option('display.max_rows', None)
    feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embedded_lr_support,
                                    'Random Forest':embedded_rf_support, 'LightGBM':embedded_lgbm_support})
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    best_features = feature_selection_df[feature_selection_df['Total'] ==6]['Feature']
    #### Your Code ends here
    return best_features



dataset_path = input('Enter the path\n') 
n = int(input("Enter number of methods"))
num_list = list(num for num in input("Enter the methods ").strip().split())[:n]
best_features = autoFeatureSelector(dataset_path, methods=num_list)
print(best_features)






