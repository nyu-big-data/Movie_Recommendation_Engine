import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k
from lightfm import LightFM
import pandas as pd
import scipy.sparse as sp

def get_df_matrix_mappings(df, row_name, col_name):
    """Map entities in interactions df to row and column indices
    Parameters
    ----------
    df : DataFrame
        Interactions DataFrame.
    row_name : str
        Name of column in df which contains row entities.
    col_name : str
        Name of column in df which contains column entities.
    Returns
    -------
    rid_to_idx : dict
        Maps row ID's to the row index in the eventual interactions matrix.
    idx_to_rid : dict
        Reverse of rid_to_idx. Maps row index to row ID.
    cid_to_idx : dict
        Same as rid_to_idx but for column ID's
    idx_to_cid : dict
    """


    # Create mappings
    rid_to_idx = {}
    idx_to_rid = {}
    for (idx, rid) in enumerate(df[row_name].unique().tolist()):
        rid_to_idx[rid] = idx
        idx_to_rid[idx] = rid

    cid_to_idx = {}
    idx_to_cid = {}
    for (idx, cid) in enumerate(df[col_name].unique().tolist()):
        cid_to_idx[cid] = idx
        idx_to_cid[idx] = cid

    return rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid


def df_to_matrix(df, row_name, col_name):
    """Take interactions dataframe and convert to a sparse matrix
    Parameters
    ----------
    df : DataFrame
    row_name : str
    col_name : str
    Returns
    -------
    interactions : sparse csr matrix
    rid_to_idx : dict
    idx_to_rid : dict
    cid_to_idx : dict
    idx_to_cid : dict
    """

    rid_to_idx, idx_to_rid,\
        cid_to_idx, idx_to_cid = get_df_matrix_mappings(df,
                                                        row_name,
                                                        col_name)

    def map_ids(row, mapper):
        return mapper[row]

    I = df[row_name].apply(map_ids, args=[rid_to_idx]).to_numpy()
    J = df[col_name].apply(map_ids, args=[cid_to_idx]).to_numpy()
    V = np.ones(I.shape[0])
    interactions = sp.coo_matrix((V, (I, J)), dtype=np.float64)
    interactions = interactions.tocsr()
    return interactions, rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid

def main():

    train = pd.read_csv(f'training_data.csv', names=['userId', 'movieId', 'rating', 'timestamp', 'split'])

    val = pd.read_csv(f'validation_data.csv', names=['userId', 'movieId', 'rating', 'timestamp', 'split'])

    test = pd.read_csv(f'test_data.csv', names=['userId', 'movieId', 'rating', 'timestamp', 'split'])

    likes, uid_to_idx, idx_to_uid,\
    mid_to_idx, idx_to_mid = df_to_matrix(train, 'userId', 'movieId')

    likes2, uid_to_idx, idx_to_uid,\
    mid_to_idx, idx_to_mid = df_to_matrix(test, 'userId', 'movieId')

    model = LightFM(loss='warp')
    # Initialize model.
    model.fit(likes, epochs=10)
    train_precision = precision_at_k(model, likes2, k=100).mean()
    print(train_precision)

if __name__ == "__main__":

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function
    main()