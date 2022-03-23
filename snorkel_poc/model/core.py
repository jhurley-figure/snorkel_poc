import os
from os.path import join, basename
from typing import Any, Dict, List

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

MODEL_PATH = "gs://jdh-bucket/projects/snorkel_pos/models"

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
        u.pickle_dump(std_scaler, join(MODEL_PATH, 'sclr.scl'))
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
            "objective": 'binary:logistic', 
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
        u.pickle_dump(mdl, join(MODEL_PATH, 'gbm.mdl'))
        return mdl

    def rf_model(self) -> RandomForestClassifier:
        X, y = self.return_data(dset='model_tr', islinear=False)
        mdl = RandomForestClassifier(max_depth=5, n_estimators=10, n_jobs=-1)
        X = X.fillna(0)
        mdl.fit(X,y)
        u.pickle_dump(mdl, join(MODEL_PATH, 'rf.mdl'))
        return mdl

    def logreg_model(self) -> LogisticRegression:
        X, y = self.return_data(dset='model_tr', islinear=True)
        mdl = LogisticRegression(C=1e8)
        mdl.fit(X,y)
        u.pickle_dump(mdl, join(MODEL_PATH, 'log.mdl'))
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


class LabelMatrixMgr():
    def __init__(self, serve_path: str, output_path: str, serve_actual_path: str):
        self.serve_path = serve_path
        self.serve_actual_path = serve_actual_path
        self.output_path = output_path
        self.mdl_dict = self.load_models()
        self.mdl_artf = self.load_artifacts()
        self.feature_cols = self.get_cols()

    def load_artifacts(self):
        res = {}
        res['sclr'] = u.pickle_load("gs://jdh-bucket/projects/snorkel_pos/models/sclr.scl")
        return res

    def load_models(self):
        res = {}
        res['gbm'] = u.pickle_load("gs://jdh-bucket/projects/snorkel_pos/models/gbm.mdl")
        res['rf'] = u.pickle_load("gs://jdh-bucket/projects/snorkel_pos/models/rf.mdl")
        res['logit'] = u.pickle_load("gs://jdh-bucket/projects/snorkel_pos/models/log.mdl")
        return res

    def get_cols(self):
        return list(self.mdl_dict['rf'].feature_names_in_)

    def create_lbl_matrix_for_srv(self):
        pq_paths = u.get_parquet_paths(self.serve_path)

        lbls = ray.get([
            process_srv_partition.remote(
                input_path=pq_path, 
                output_path=self.output_path, 
                actual_path=self.serve_actual_path, 
                cols_to_use=self.feature_cols, 
                mdl_dict=self.mdl_dict,
                artf_dict=self.mdl_artf
            ) 
            for pq_path in pq_paths
        ])
        return lbls

@ray.remote
def process_srv_partition(
        input_path: str, 
        output_path: str, 
        actual_path: str, 
        cols_to_use: List[str], 
        mdl_dict: Dict[str, Any],
        artf_dict: Dict[str, Any]
):
    bname = basename(input_path)
    columns = cols_to_use + ['encrypted_nb', 'applied']
    df = u.read_parquet_pd(input_path, columns=columns)
    df = df.reset_index().set_index(['record_nb', 'encrypted_nb'])
    #df['applied'].to_parquet(actual_path)
    df = df.drop('applied', axis=1)
    res = {}
    for m in mdl_dict:
        if m in ['rf', 'logit']:
            df = df.fillna(0)
            if m == 'logit':
                df = pd.DataFrame(artf_dict['sclr'].transform(df), columns=df.columns, index=df.index)
            pred = pd.Series(mdl_dict[m].predict_proba(df)[:,1], index=df.index)
            pred2 = pd.Series(mdl_dict[m].predict(df), index=df.index)
        else:
            dmat = xgb.DMatrix(df)
            pred = pd.Series(mdl_dict[m].predict(dmat), index=df.index)
            pred2 = None
        res[m] = pred
        if pred2 is not None:
            res['%s_pred' %m] = pred2
    res = pd.DataFrame(res)

    columns = [f'F{x}' for x in range(15)]
    biz_rules = pd.DataFrame(get_rules(df), columns=columns, index=df.index)
    res = pd.concat([res, biz_rules], axis=1)

    res.to_parquet(join(output_path, bname))
    
def get_rules(df: pd.DataFrame) -> np.ndarray:
    res = (apply_rule1(df), apply_rule2(df), apply_rule3(df),
        apply_rule4(df), apply_rule5(df), apply_rule6(df),
        apply_rule7(df), apply_rule8(df), apply_rule9(df),
        apply_rule10(df), rule_fip5020(df), rule_fip8120(df),
        rule_iqb9410(df), rule_iqf9510(df), rule_iqt9526(df)
    )
    return np.concatenate(res, axis=1)

def convert_rule_with_abstain(mask: np.ndarray, val = int) -> np.ndarray:
    mask = np.array(mask).reshape(-1, 1)
    res = (np.zeros(len(mask)) - 1).reshape(-1, 1)
    res[mask] = val
    return res

def convert_rule_to_binary(mask: np.ndarray, val = int) -> np.ndarray:
    mask = np.array(mask).reshape(-1, 1)
    if val == 1:
        res = (np.zeros(len(mask))).reshape(-1, 1)
    else:
        res = (np.ones(len(mask))).reshape(-1, 1)
    res[mask] = val
    return res


def rule_fip5020(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip5020'] > 4589)
    return convert_rule_to_binary(mask, 1)

def rule_fip8120(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip8120'] > 0.5)
    return convert_rule_to_binary(mask, 1)

def rule_iqt9526(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_iqt9526'] > 0.5)
    return convert_rule_to_binary(mask, 1)

def rule_iqf9510(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_iqf9510'] > 0.5)
    return convert_rule_to_binary(mask, 1)

def rule_iqb9410(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_iqb9410'] > 2.5)
    return convert_rule_to_binary(mask, 1)

def apply_rule1(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip5020'] > 4589) & \
        (df['premier_v1_2_iqb9410'] > 2.5)
        #(df['premier_v1_2_fip8120'] <= 6.5)
    return convert_rule_with_abstain(mask, 1)

def apply_rule3(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip5020'] <= 4589) & \
        (df['premier_v1_2_iqt9526'] <= 0.5)
    return convert_rule_with_abstain(mask, 0)

def apply_rule2(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip8120'] > 0.5) & \
            (df['premier_v1_2_iqb9410'] > 2.5) & \
            (df['premier_v1_2_iqf9510'] > 0.5)
    return convert_rule_to_binary(mask, 1)


def apply_rule4(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip5020'] > 4888.5) & \
            (df['premier_v1_2_iqt9526'] > 8.5) & \
            (df['ficeclv9_score'] > 683)
    return convert_rule_with_abstain(mask, 1)


def apply_rule5(df: pd.DataFrame) -> np.ndarray:
    #mask = (df['premier_v1_2_fip5020'] > 4588) & \
    mask = (df['premier_v1_2_pil0438'] > 1.5)
    return convert_rule_to_binary(mask, 1)


def apply_rule6(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip5020'] > 4589)
    return convert_rule_to_binary(mask, 1)


def apply_rule7(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip8120'] <= 6.5) & \
    (df['premier_v1_2_iqb9410'] > 2.5)
    return convert_rule_with_abstain(mask, 1)


def apply_rule8(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip5020'] > 4588) & \
    (df['ficeclv9_score'] <= 680)
    return convert_rule_to_binary(mask, 1)


def apply_rule9(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip5020'] <= 4588.5) & \
    (df['premier_v1_2_iqb9410'] <= 1.5) & \
    (df['premier_v1_2_iqt9526'] <= 0.5)
    return convert_rule_with_abstain(mask, 0)


def apply_rule10(df: pd.DataFrame) -> np.ndarray:
    mask = (df['premier_v1_2_fip8120'] <= 0.5) & \
    (df['ficeclv9_score'] <= 686) & \
    (df['premier_v1_2_iqt9526'] <= 0.5)
    return convert_rule_to_binary(mask, 0)

if __name__ == "__main__":
    #poc = SnorkelDataPrepper(campaign='37')
    lmm = LabelMatrixMgr(serve_path='gs://jdh-bucket/projects/snorkel_pos/data/serve/2109A.parquet',
            serve_actual_path='gs://jdh-bucket/projects/snorkel_pos/data/serve_actuals/2109A.parquet',
            output_path='gs://jdh-bucket/projects/snorkel_pos/data/serve_lbls/2109A'
    )
    lmm.create_lbl_matrix_for_srv()
    #u.pickle_dump(poc.label_matrix(), "gs://jdh-bucket/projects/snorkel_pos/lbl_matrix.mtx")
    #u.pickle_dump(poc.dmatrices, "gs://jdh-bucket/projects/snorkel_pos/dmatrics.dict")
    #smgr = SnorkelMgr(poc.label_matrix()) 

