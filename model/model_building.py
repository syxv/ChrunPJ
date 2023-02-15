#import libraries
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("./Bank Customer Churn Prediction.csv")
df=df.drop('customer_id',axis=1)
class Dummy_Transformer(object):
    
    def fit(self, X, y=None):
        self.keys = set(X)
    
    def transform(self, X, y=None):
        res = {}
        for key in self.keys:
            res[key] = [0]*len(X)    
        for i, item in enumerate(X):
            if item in self.keys:
                res[item][i] = 1
        return pd.DataFrame(res)
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)  
dummytf = Dummy_Transformer()

frames = [df,dummytf.fit_transform(df.country)]
dfNew=pd.concat(frames,axis=1,join='inner')
dfNew["gender"] = LabelEncoder().fit_transform(dfNew["gender"])
# move the column to end of list using index, pop and insert
columns = list(dfNew)
columns.insert(900, columns.pop(columns.index('churn')))
dfNew = dfNew.loc[:, columns]
dfNum=dfNew.drop('country',axis=1)

features = list(dfNum.columns)
target = "churn"
features.remove(target)

X = dfNum[features]
y = dfNum[target]

#print(X.columns)

steps = [('rescale', StandardScaler()),
         ('dt',lgb.LGBMClassifier())]
clfLgb = Pipeline(steps)
clfLgb.fit(X, y)


with open('model.pkl', 'wb') as f:
    pickle.dump(clfLgb, f)
