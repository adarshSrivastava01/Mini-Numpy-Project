import numpy as np

# DATA NORMALIZATION SECTION

X = np.random.randint(5001,size=(1000,20)) #GENERATING A RANDOM ARRAY

# print(X.shape)

ave_cols = np.mean(X, axis=0) # AVERAGE VALUE

std_cols = np.std(X, axis=0) # STANDARD DEVIATION

# print(ave_cols.shape)
# print(std_cols.shape)

X_norm = (X - ave_cols) / std_cols  # NORMALIZED DATA

# print(X_norm.mean())
# print(np.amin(X_norm, axis=0))
# print(np.amax(X_norm, axis=0))


# DATA SEPERATION

row_indices = np.random.permutation(X_norm.shape[0])

train_index = row_indices[:600]
crossvalid_index = row_indices[600:800]
test_index = row_indices[800:]

# print(train_index)
# print(crossvalid_index)
# print(test_index)

X_train = X_norm[train_index, :]
X_crossVal = X_norm[crossvalid_index, :]
X_test = X_norm[test_index, :]

print(X_train.shape)
print(X_crossVal.shape)
print(X_test.shape)