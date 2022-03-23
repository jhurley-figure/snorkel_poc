from os.path import join, basename
import re

import pandas as pd
import ray

import bqutil.core as bq
import dmnet.util as util

def create_parquet_file(
        campaign: str = '2109A',
        output_path: str = "gs://jdh-bucket/projects/snorkel_pos/data/apps") -> None:
    ssql = f"""
    SELECT distinct campaign, record_nb, product_type
    FROM `figure-production.reporting.lkup_dm_applications`
    WHERE campaign in ('{campaign}') and product_type in ('HELOC')
    """
    output = join(output_path, f'{campaign}.parquet')
    df = bq.read_sql(ssql)
    df.to_parquet(output)
    return bq.read_sql(ssql)

@ray.remote
def process_partition(pq_path: str, app_file: pd.DataFrame, output_path: str) -> None:
    df = util.read_parquet_pd(pq_path)
    df = df.set_index('record_nb')

    bname = basename(pq_path)
    df = df.merge(app_file, how='left', left_index=True, right_index=True)
    df['applied'] = df['applied'].fillna(0)
    
    output_path = join(output_path, bname)
    df.to_parquet(output_path)

def update_serve_data(
    campaign: str = '2109A'
) -> None:

    serve_data_path = "gs://dmnet/helocnet/campaign/37/dmatrix/serve.parquet"
    srv_pqt_path = util.get_parquet_paths(serve_data_path)
    
    app_df = util.read_parquet_pd("gs://jdh-bucket/projects/snorkel_pos/data/apps/2109A.parquet")
    app_df['applied'] = 1
    app_df = app_df[['applied', 'record_nb']]
    app_df = ray.put(app_df.set_index('record_nb'))
    
    output_path = ray.put("gs://jdh-bucket/projects/snorkel_pos/data/serve/2109A.parquet")

    ray.get([
        process_partition.remote(path, app_df, output_path) for path in srv_pqt_path
    ])


if __name__ == "__main__":
    util.init_ray(distributed= False)

    # create_parquet_file()
    update_serve_data()
