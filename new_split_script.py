import pandas as pd

names = ['userId', 'movieId', 'rating', 'timestamp']
ratings = pd.read_csv(f'ratings_small.csv', names=names)

val_df = ratings.groupby("userId").sample(n=5, replace=False)

ratings = ratings.merge(val_df, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only']
ratings = ratings.drop('_merge', axis=1)

test_df = ratings.groupby("userId").sample(n=5, replace=False)

train_df = ratings.merge(test_df, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only']
train_df = train_df.drop('_merge', axis=1)

train_df.to_csv('new_train_small.csv', index=False)
val_df.to_csv('new_val_small.csv', index=False)
test_df.to_csv('new_test_small.csv', index=False)
