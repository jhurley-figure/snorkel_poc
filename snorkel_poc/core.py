import numpy as np
import os
from os.path import join

import pandas as pd
import ray

from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel.preprocess import preprocessor

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import dmnet.util as u

MODEL_LOC = "gs://jdh-bucket/projects/snorkel_pos/rf.mdl"

class SnorkelPOC():

    def __init__(self, campaign: str = '42.prod'):
        self.campaign = campaign
        self.dmatrices = self.get_data()
        #self.std_scaler = self.get_std_scaler()
        self.rf = self.rf_model()
        #self.logreg = self.logreg_model()


    def get_data(self):
        df = self.load_data()
        X, y = df.drop(['y'], axis=1), df['y']
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=1000000, test_size=100000, stratify=y)
        return {'for_testing': pd.concat([X_tr, y_tr], axis=1),
                'for_label_making': pd.concat([X_te, y_te], axis=1)
        }

    def load_data(self) -> pd.DataFrame:
        fname = f"gs://dmnet/heloc/campaign/{self.campaign}/dmatrix/tr_gbm.parquet"
        return pd.read_parquet(fname)

    def rf_model(self) -> RandomForestClassifier:
        mdl = RandomForestClassifier(max_depth=5, n_estimators=10)
        X, y = self.dmatrices['for_label_making'].drop('y', axis=1), self.dmatrices['for_label_making']['y']
        X = X.fillna(0)
        mdl.fit(X,y)
        u.pickle_dump(mdl, MODEL_LOC)
        return mdl

    def get_std_scaler(self) -> StandardScaler:
        sclr = StandardScaler()
        X = self.dmatrices['for_label_making'].drop('y', axis=1).fillna(0)
        sclr.fit(X)
        return sclr

    def logreg_model(self) -> LogisticRegression:
        mdl = LogisticRegression(C=1e8)
        X, y = self.dmatrices['for_label_making'].drop('y', axis=1), self.dmatrices['for_label_making']['y']
        X = X.fillna(0)
        X = self.std_scaler.transform(X)
        mdl.fit(X,y)
        return mdl

    def label_function(self):
        X = self.dmatrices['for_testing'].drop(['y'], axis=1)
        rf_preds = pd.Series(self.rf.predict(X.fillna(0)))
        cltv_flag = pd.Series((X['cltv'] > 0.3) * 1).astype(int)
        lbl_mtrx = pd.concat([rf_preds, cltv_flag], axis=1).to_numpy()
        return lbl_mtrx

@labeling_function()
def cltv_bk_pt(x):
    x = 0 if np.isnan(x['cltv']) else x['cltv']
    return 1 if x > 0.3 else 0

if __name__ == "__main__":
    poc = SnorkelPOC(campaign='42.prod')
    u.pickle_dump(poc.label_function(), "gs://jdh-bucket/projects/snorkel_pos/test_lbl.fnc")
