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
        self.std_scaler = self.create_std_scaler()
        self.dmatrices_for_linear = self.transform_for_linear()
        self.rf = self.rf_model()
        self.xgb = self.gbm_model()
        self.logreg = self.logreg_model()
    
    def load_data(self) -> pd.DataFrame:
        fname = f"gs://dmnet/heloc/campaign/{self.campaign}/dmatrix/tr_gbm.parquet"
        res = pd.read_parquet(fname).set_index(['record_nb', 'encrypted_nb'])
        return res

    def get_data(self):
        df = self.load_data()
        X, y = df.drop(['y'], axis=1), df['y']
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=1000000, test_size=100000, stratify=y)
        X_mdl, _, y_mdl, _ = train_test_split(X, y, train_size=100000, stratify=y)
        return {'snorkel_tr': pd.concat([X_tr, y_tr], axis=1),
                'snorkel_te': pd.concat([X_te, y_te], axis=1),
                'model_tr': pd.concat([X_mdl, y_mdl], axis=1)
        }

    def create_std_scaler(self) -> StandardScaler:
        std_scaler = StandardScaler()
        X = self.dmatrices['model_tr'].drop('y', axis=1).fillna(0)
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
    
    def return_data(self, dset: str, islinear: bool=True):
        if islinear:
            X = self.dmatrices_for_linear[dset].drop('y', axis=1)
            y = self.dmatrices_for_linear[dset]['y']
        else:
            X = self.dmatrices[dset].drop('y', axis=1)
            y = self.dmatrices[dset]['y']
        return X, y

    def gbm_model(self) -> xgb.DMatrix:
        X, y = self.return_data(dset='model_tr', islinear=False)
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
        X, y = self.return_data(dset='model_tr', islinear=False)
        mdl = RandomForestClassifier(max_depth=5, n_estimators=10, n_jobs=-1)
        X = X.fillna(0)
        mdl.fit(X,y)
        return mdl

    def logreg_model(self) -> LogisticRegression:
        X, y = self.return_data(dset='model_tr', islinear=True)
        mdl = LogisticRegression(C=1e8)
        mdl.fit(X,y)
        return mdl

    def label_matrix(self, istrain:bool= True):
        if istrain:
            X, y = self.return_data(dset='snorkel_tr', islinear=False)
        else:
            X, y = self.return_data(dset='snorkel_te', islinear=False)
        rf_preds = self.rf.predict_proba(X.fillna(0))[:, 1].reshape(-1,1)
        xgb_X = xgb.DMatrix(X.fillna(0))
        xgb_preds = self.xgb.predict(xgb_X).reshape(-1,1)
        
        if istrain:
            X, y = self.return_data(dset='snorkel_tr', islinear=True)
        else:
            X, y = self.return_data(dset='snorkel_te', islinear=True)
        logreg_preds = self.logreg.predict_proba(X.fillna(0))[:, 1].reshape(-1, 1)
        #knn_preds = self.knn.predict_proba(X.fillna(0))[:, 1].reshape(-1, 1)
        #print ('done w knn')
        other_rules = get_rules(X)

        mdls = (rf_preds, xgb_preds, logreg_preds, other_rules)

        return np.concatenate(mdls, axis=1)

class SnorkelMgr():
    def __init__(self, lbl_matrix, mtrx_lbls=None):
        self.lbl_matrix = self.convert_to_class(lbl_matrix)
        self.mtrx_lbls = mtrx_lbls
        self.ub = 65
        self.lb = 40

    def convert_to_class(self, mtx):
        upr_pctl = np.percentile(mtx, q=self.ub, axis=0)
        lwr_pctl = np.percentile(mtx, q=self.lb, axis=0)
        mtx[mtx > upr_pctl] = 1
        mtx[mtx < lwr_pctl] = -1
        mtx[(mtx <= upr_pctl) & (mtx >= lwr_pctl)] = 0
        return mtx

    def lbl_analysis(self):
        LFAnalysis(L=self.lbl_matrix, lfs=self.mtrx_lbls).lf_summary()

    def train_snorkel_model(self):
        lbl_mdl = LabelModel(cardinality=2, verbose=True)
        lbl_mdl.fit(L_train=self.lbl_matrix, n_epochs=100, log_freq=100, seed=123)

   
def get_rules(df: pd.DataFrame) -> np.ndarray:
    res = (apply_rule1(df), apply_rule2(df), apply_rule3(df), 
        apply_rule4(df), apply_rule5(df), apply_rule6(df), 
        apply_rule7(df), apply_rule8(df), apply_rule9(df), apply_rule10(df)
    )
    return np.concatenate(res, axis=1)

def apply_rule1(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip5020'] > 2825) & \
    (df['premier_v1_2_bca6220'] > 0.5) & \ 
    (df['premier_v1_2_iqt9526'] <= 6.5)
    return np.array(mask.astype(int)).reshape(-1, 1)


def apply_rule2(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_iqb9410'] > 1.5) & \
    (df['premier_v1_2_iqt9526'] > 0.5) & \
    (df['ficeclv9_score' > 692] > 692)
    return np.array(mask.astype(int)).reshape(-1, 1)


def apply_rule3(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip5020'] <= 2825) & \
    (df['premier_v1_2_iqb9410'] > 2.5) & \
    (df['premier_v1_2_iqt9526'] > 0.5)
    return np.array(mask.astype(int)).reshape(-1, 1)


def apply_rule4(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip8120'] <= 0.5) & \
    (df['premier_v1_2_bca6220'] <= 0.5) & \
    (df['premier_v1_2_iqt9526'] <= 0.5)
    return np.array(~mask.astype(int)).reshape(-1, 1)


def apply_rule5(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip8120'] > 0.5) & \
    (df['premier_v1_2_fip5020'] <= 6354) & \
    (df['premier_v1_2_iqt9526'] > 0.5)
    return np.array(mask.astype(int)).reshape(-1, 1)


def apply_rule6(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip8120'] <= 0.5) & \
    (df['premier_v1_2_bca6220'] <= 0.5) & \
    (df['premier_v1_2_iqt9526'] <= 0.5)
    return np.array(~mask.astype(int)).reshape(-1, 1)


def apply_rule7(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip5020'] <= 2825) & \
    (df['premier_v1_2_bca6220'] <= 0.5) & \
    (df['premier_v1_2_iqm9540'] <= 0.5)
    return np.array(~mask.astype(int)).reshape(-1, 1)


def apply_rule8(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip8120'] > 0.5) & \
    (df['premier_v1_2_fip5020'] > 5855) & \
    (df['premier_v1_2_iqt9526'] <= 0.5)
    return np.array(mask.astype(int)).reshape(-1, 1)


def apply_rule9(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip8120'] > 0.5) & \
    (df['ficeclv9_score'] > 692) & \
    (df['premier_v1_2_iqt9526'] > 0.5)
    return np.array(mask.astype(int)).reshape(-1, 1)


def apply_rule10(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip5020'] > 2825) & \
    (df['ficeclv9_score'] <= 709) & \
    (df['premier_v1_2_iqt9526'] <= 5.5)
    return np.array(mask.astype(int)).reshape(-1, 1)

if __name__ == "__main__":
    poc = SnorkelDataPrepper(campaign='42.prod')
    u.pickle_dump(poc.label_matrix(), "gs://jdh-bucket/projects/snorkel_pos/lbl_matrix.mtx")
    u.pickle_dump(poc.dmatrices, "gs://jdh-bucket/projects/snorkel_pos/dmatrics.dict")
    #smgr = SnorkelMgr(poc.label_matrix()) 
