import os
import glob
import torch
import torchvision
import numpy as np
import pandas as pd
import tifffile
import pyarrow.parquet as pq
import dask.dataframe as dd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


import pyarrow.parquet as pq
import pyarrow as pa

chunk_size = 10000
parquet_file = pq.ParquetFile('/rds/general/user/abm1818/projects/ssa_satellite_imagery/ephemeral/data/NC_emb_sub_satlas_masked.parquet')
# 1st option
""" In writing part we are writing chunk by chunk after processing
at output path it will create directory with .parquet suffix and
part files will be created in it. 
In this approach iterating add some performance hit"""
for batch in parquet_file.iter_batches(batch_size=chunk_size):
    table = pa.Table.from_batches([batch])
    # Process the chunk (table)
    # Joining data with another small file dataframe
    chunk_df = table.to_pandas(split_blocks=True, self_destruct=True)
    chunk_df = chunk_df.merge(small_file_df, on=join_columns, how=join_type,
                            suffixes=('_left', '_right')
    final_table = pa.Table.from_pandas(chunk_df)
    pq.write_to_dataset(final_table, root_path=output_path,
                        partition_cols=partition_key_list)


#df_fe = pd.read_parquet('/rds/general/user/abm1818/projects/ssa_satellite_imagery/ephemeral/data/NC_emb_sub_satlas_masked.parquet', engine='fastparquet')
#df = pd.read_parquet('data.parquet', engine='fastparquet')
#df_fe = dd.read_parquet('/rds/general/user/abm1818/projects/ssa_satellite_imagery/ephemeral/data/NC_emb_sub_satlas_masked.parquet', engine='pyarrow')
print('read')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_fe)
    
pca = PCA(n_components=100)
pri_comp = pca.fit_transform(X_scaled)


explained_variance_ratio = pca.explained_variance_ratio_
print(explained_variance_ratio.cumsum())

pc_df = pd.DataFrame(pri_comp)
print('saved as df')

pc_df.to_csv('/rds/general/user/abm1818/projects/ssa_satellite_imagery/ephemeral/data/NC_emb_sub_satlas_masked_PC100.parquet')
print('done')
