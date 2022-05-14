import numpy as np
from lightfm.data import Dataset
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

def train_test_split(interactions, split_count, fraction=None):
    """
    Split recommendation data into train and test sets
    Params
    ------
    interactions : scipy.sparse matrix
        Interactions between users and items.
    split_count : int
        Number of user-item-interactions per user to move
        from training to test set.
    fractions : float
        Fraction of users to split off some of their
        interactions into test set. If None, then all
        users are considered.
    """
    # Note: likely not the fastest way to do things below.
    train = interactions.copy().tocoo()
    val = sp.lil_matrix(train.shape)
    test = sp.lil_matrix(train.shape)

    if fraction:
        try:
            user_index = np.random.choice(
                np.where(np.bincount(train.row) >= split_count * 2)[0],
                replace=False,
                size=np.int64(np.floor(fraction * train.shape[0]))
            ).tolist()
        except:
            print(('Not enough users with > {} '
                  'interactions for fraction of {}')\
                  .format(2*split_count, fraction))
            raise
    else:
        user_index = range(train.shape[0])

    train = train.tolil()

    for user in user_index:
        val_test_interactions = np.random.choice(interactions.getrow(user).indices,
                                        size=split_count,
                                        replace=False)
        split = np.array_split(val_test_interactions, 2)
        val_interactions = split[0]
        test_interactions = split[1]
        train[user, val_test_interactions] = 0.
        # These are just 1.0 right now
        val[user, val_interactions] = interactions[user, val_interactions]
        test[user, test_interactions] = interactions[user, test_interactions]


    # Test and training are truly disjoint
    assert(train.multiply(test).nnz == 0)
    return train.tocsr(), val.tocsr(), test.tocsr(), user_index

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

    ratings = pd.read_csv(f'ratings_small.csv', names=['userId', 'movieId', 'rating', 'timestamp', 'split'])
    train_df = pd.read_csv(f'training_data.csv', names=['userId', 'movieId', 'rating', 'timestamp', 'split'])
    val_df = pd.read_csv(f'validation_data.csv', names=['userId', 'movieId', 'rating', 'timestamp', 'split'])
    test_df = pd.read_csv(f'test_data.csv', names=['userId', 'movieId', 'rating', 'timestamp', 'split'])

    ratings = ratings[['userId','movieId']]
    train_df = train_df[['userId', 'movieId']]
    val_df = val_df[['userId', 'movieId']]
    test_df = test_df[['userId', 'movieId']]

    likes, uid_to_idx, idx_to_uid, mid_to_idx, idx_to_mid = df_to_matrix(ratings, 'userId', 'movieId')

    train = subset_to_matrix(likes, uid_to_idx, mid_to_idx, ratings, train_df)
    val = subset_to_matrix(likes, uid_to_idx, mid_to_idx, ratings, val_df)
    test = subset_to_matrix(likes, uid_to_idx, mid_to_idx, ratings, test_df)
    
    model = LightFM(loss='warp')
    # Initialize model.
    model.fit(train, epochs=10)
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