import numpy as np
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k
from lightfm import LightFM
import pandas as pd
import scipy.sparse as sp

def get_df_matrix_mappings(df, row_name, col_name):
    """
    Obtain mappings of interactions between row (matrix) item and column (matrix) item. 

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
        Mapping from Row IDs -> Rows Indices 
    idx_to_rid : dict
        Mapping from Rows Indices -> Row IDs
    cid_to_idx : dict
        Mapping from Column IDs -> Column Indices 
    idx_to_cid : dict
        Mapping from Column Indices -> Column IDs

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
    """
    Obtains the sparse matrix with interaction from the dataframe

    Parameters
    ----------
    df : DataFrame
    row_name : str
    col_name : str

    Returns
    -------
    interactions : Sparse Matrix
    rid_to_idx : dict
        Mapping from Row IDs -> Rows Indices 
    idx_to_rid : dict
        Mapping from Rows Indices -> Row IDs
    cid_to_idx : dict
        Mapping from Column IDs -> Column Indices 
    idx_to_cid : dict
        Mapping from Column Indices -> Column IDs
    """

    rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid = get_df_matrix_mappings(df, row_name, col_name)

    def map_ids(row, mapper):
        return mapper[row]

    I = df[row_name].apply(map_ids, args=[rid_to_idx]).to_numpy()
    J = df[col_name].apply(map_ids, args=[cid_to_idx]).to_numpy()
    V = np.ones(I.shape[0])
    interactions = sp.coo_matrix((V, (I, J)), dtype=np.float64)
    interactions = interactions.tocsr()
    return interactions, rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid

def subset_to_matrix(interactions, uid_to_idx, mid_to_idx, ratings, subset):

    diff = ratings.merge(subset, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only']
    user_list = diff['userId'].values
    movie_list = diff['movieId'].values

    sub_mat = interactions.copy().tolil()
    
    for user,movie in zip(user_list,movie_list):
        uidx = uid_to_idx[user]
        midx = mid_to_idx[movie]

        sub_mat[uidx, midx] = 0.
    
    return sub_mat.tocsr()

def main():
    ## Read in the CSV files
    colnames=['userId', 'movieId', 'rating', 'timestamp']
    ratings = pd.read_csv(f'ratings_small.csv', names=colnames)
    train_df = pd.read_csv(f'new_train_small.csv')
    val_df = pd.read_csv(f'new_val_small.csv')
    test_df = pd.read_csv(f'new_test_small.csv')

    train_df.reset_index(inplace=True)

    print(ratings.head())
    print(train_df.head())

    ## Select only the User and Movie column that we would need
    ratings = ratings[['userId','movieId']]
    train_df = train_df[['userId', 'movieId']]
    val_df = val_df[['userId', 'movieId']]
    test_df = test_df[['userId', 'movieId']]

    likes, uid_to_idx, idx_to_uid, mid_to_idx, idx_to_mid = df_to_matrix(ratings, 'userId', 'movieId')

    ## Obtain Sparse Matrices with interactions
    train = subset_to_matrix(likes, uid_to_idx, mid_to_idx, ratings, train_df)
    val = subset_to_matrix(likes, uid_to_idx, mid_to_idx, ratings, val_df)
    test = subset_to_matrix(likes, uid_to_idx, mid_to_idx, ratings, test_df)

    assert(train.multiply(val).nnz == 0)
    assert(train.multiply(test).nnz == 0)
    assert(val.multiply(test).nnz == 0)
    
    ## Initialize model
    model = LightFM(loss='warp')
    
    ## Train the model
    model.fit(train, epochs=10)

    ## Compute Precision@K
    train_precision = precision_at_k(model, train, k=100).mean()
    val_precision = precision_at_k(model, val, k=100).mean()
    test_precision = precision_at_k(model, test, k=100).mean()

    
    print(train_precision)
    print(val_precision)
    print(test_precision)

if __name__ == "__main__":

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function
    main()
