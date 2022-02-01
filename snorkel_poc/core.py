import os
from os.path import join

import numpy as np
import pandas as pd
import ray

from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis, filter_unlabeled_dataframe
from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from snorkel.preprocess import preprocessor

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import dmnet.util as u

MODEL_LOC = "gs://jdh-bucket/projects/snorkel_pos/rf.mdl"

class SnorkelDataPrepper():

    def __init__(self, campaign: str = '42.prod'):
        self.campaign = campaign
        self.dmatrices = self.get_data()
        self.std_scaler = self.std_scaler()
        self.dmatrices_for_linear = self.transform_for_linear()
        self.rf = self.rf_model()
        self.xgb = self.gbm_model()
        self.knn = self.knn_model()
        self.logreg = self.logreg_model()
    
    def load_data(self) -> pd.DataFrame:
        fname = f"gs://dmnet/heloc/campaign/{self.campaign}/dmatrix/tr_gbm.parquet"
        res = pd.read_parquet(fname).set_index(['record_nb', 'encrypted_nb'])
        return res

    def get_data(self):
        df = self.load_data()
        X, y = df.drop(['y'], axis=1), df['y']
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=1000000, test_size=100000, stratify=y)
        return {'for_testing': pd.concat([X_tr, y_tr], axis=1),
                'for_label_making': pd.concat([X_te, y_te], axis=1)
        }

    def std_scaler(self) -> StandardScaler:
        std_scaler = StandardScaler()
        X = self.dmatrices['for_label_making'].drop('y', axis=1).fillna(0)
        std_scaler.fit(X)
        return std_scaler
        
    def transform_for_linear(self):
        res = {}
        for part in self.dmatrices:
            X = self.dmatrices[part].drop('y', axis=1).fillna(0).copy()
            y = self.dmatrices[part]['y']
            X = pd.DataFrame(self.std_scaler.transform(X), columns=X.columns, index=X.index)
            Xy = pd.concat([X, y], axis=1)
            res[part] = Xy
        return res
    
    def return_data(self, istrain:bool=True, islinear: bool=True):
        if istrain and islinear:
            X = self.dmatrices_for_linear['for_label_making'].drop('y', axis=1)
            y = self.dmatrices_for_linear['for_label_making']['y']
        elif not istrain and islinear:
            X = self.dmatrices_for_linear['for_testing'].drop('y', axis=1)
            y = self.dmatrices_for_linear['for_testing']['y']
        elif istrain and not islinear:
            X = self.dmatrices['for_label_making'].drop('y', axis=1)
            y = self.dmatrices['for_label_making']['y']
        else:
            X = self.dmatrices['for_testing'].drop('y', axis=1)
            y = self.dmatrices['for_testing']['y']
        return X, y

    def knn_model(self):
        X, y = self.return_data(istrain=True, islinear=True)
        X, _, y, _ = train_test_split(X, y, train_size=1000)
        mdl = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        mdl.fit(X,y)
        return mdl
   
    def gbm_model(self) -> xgb.DMatrix:
        X, y = self.return_data(istrain=True, islinear=False)
        Xy = xgb.DMatrix(X, y)
        params = {
            "eta": 0.1, 
            "max_leaves": 10,
            "min_child_weight": 800,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "eval_metric": "auc"
        }
        
        train_kwargs = {
            "params": params,
            "dtrain": Xy,
            "evals": [(Xy, "train")]
        }
        
        train_params = {
            "num_boost_round": 100,
            "early_stopping_rounds": 50,
            "verbose_eval": 100
        }

        mdl = xgb.train(**train_kwargs, **train_params)
        return mdl

    def rf_model(self) -> RandomForestClassifier:
        X, y = self.return_data(istrain=True, islinear=False)
        mdl = RandomForestClassifier(max_depth=5, n_estimators=10, n_jobs=-1)
        X = X.fillna(0)
        mdl.fit(X,y)
        return mdl

    def logreg_model(self) -> LogisticRegression:
        X, y = self.return_data(istrain=True, islinear=True)
        mdl = LogisticRegression(C=1e8)
        mdl.fit(X,y)
        return mdl

    def label_matrix(self):
        X, y = self.return_data(istrain=False, islinear=False)
        rf_preds = self.rf.predict(X.fillna(0)).reshape(-1,1)
        xgb_X = xgb.DMatrix(X.fillna(0))
        xgb_preds = self.xgb.predict(xgb_X).reshape(-1,1)
        xgb_preds[xgb_preds >= 0.5] = 1
        xgb_preds[xgb_preds < 0.5] = 0

        X, y = self.return_data(istrain=False, islinear=True) 
        logreg_preds = self.logreg.predict(X.fillna(0)).reshape(-1, 1)
        
        knn_preds = self.knn.predict(X.fillna(0)).reshape(-1, 1)

        mdls = (rf_preds, xgb_preds, logreg_preds, knn_preds)

        return np.concatenate(mdls, axis=1)


class SnorkelMgr():
    def __init__(self, lbl_matrix, mtrx_lbls):
        self.lbl_matrix = lbl_matrix
        self.mtrx_lbls = mtrx_lbls

    def lbl_analysis(self):
        LFAnalysis(L=self.lbl_matrix, lfs=self.mtrx_lbls).lf_summary()

    def train_snorkel_model(self):
        lbl_mdl = LabelModel(cardinality=2, verbose=True)
        lbl_mdl.fit(L_train=self.lbl_matrix, n_epochs=100, log_freq=100, seed=123)

    
if __name__ == "__main__":
    poc = SnorkelDataPrepper(campaign='42.prod')
    smgr = SnorkelMgr(poc.label_matrix(), mtrx_lbls=['rf', 'xgb', 'logit', 'knn'])
    print (smgr.lbl_analysis())
