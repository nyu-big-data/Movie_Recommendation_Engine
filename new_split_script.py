import pandas as pd

names = ['userId', 'movieId', 'rating', 'timestamp']
ratings = pd.read_csv(f'ratings_large.csv', names=names)

val_df = ratings.groupby("userId").sample(frac=0.25, replace=False)

ratings = ratings.merge(val_df, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only']
ratings = ratings.drop('_merge', axis=1)

test_df = ratings.groupby("userId").sample(frac=0.25, replace=False)

train_df = ratings.merge(test_df, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only']
train_df = train_df.drop('_merge', axis=1)

train_df.to_csv('new_train_large.csv', index=False)
val_df.to_csv('new_val_large.csv', index=False)
test_df.to_csv('new_test_large.csv', index=False)
