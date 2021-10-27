# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 09:16:24 2021

@author: PatCa
"""


from xgboost import XGBClassifier
import pickle


import cleaning_functions


#Ingest dataset, clean data, prep data. Separate text data and numeric
X_train, X_test, Y_train, Y_test, word_df, word_df_train,word_df_test, _ = cleaning_functions.PCA_Data()


#%%

# Define model
eval_set = [(X_train,Y_train),(X_test,Y_test)]
eval_metric = ['mlogloss']
clf = XGBClassifier()

# Send values to pipeline
xgb_pca = clf.fit(X_train,Y_train,
                       eval_set=eval_set, eval_metric=eval_metric,
                       early_stopping_rounds=10, verbose=True)

xgb_pred = xgb_pca.predict(X_train)
xgb_pred_test = xgb_pca.predict(X_test)



#%% Save and load trained model

file_name = "model_artifacts/xgb_pca.pkl"

pickle.dump(xgb_pca, open(file_name,'wb'))






