import os

files_to_move = os.listdir('./rgb_train_features')
for file_ in files_to_move:
    os.rename(os.path.join('./temp_feats', file_), os.path.join('./temp_rgb_train_features', file_))

files_to_move = os.listdir('./rgb_test_features')
for file_ in files_to_move:
    os.rename(os.path.join('./temp_feats', file_), os.path.join('./temp_rgb_test_features', file_))
