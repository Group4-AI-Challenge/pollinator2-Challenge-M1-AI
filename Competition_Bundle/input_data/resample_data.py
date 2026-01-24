import numpy as np
import os
from sklearn.model_selection import train_test_split
import json



X_path = os.path.join(os.path.dirname(__file__), "X.npy")
y_path = os.path.join(os.path.dirname(__file__), "y.npz")

X = np.load(X_path)
y = np.load(y_path)['y']


y = y.astype(int)

# Let's do a 0.8-0.2 train-split

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)


# Saving the test in the json
data = {
    "y_test" : y_test.tolist()
}

with open(os.path.join(os.path.dirname(__file__),  "y_test.json"), "w") as f:
    json.dump(data, f)

np.save(os.path.join(os.path.dirname(__file__),  "X_test.npy"), X_test)


print("--"*20)

print(f"X shape before : {X.shape}")
print(f"y shape before : {y.shape}")
print("--"*20)


labels = ["Neg","Fourmis","Coleoptere","Petite abeille sauvage","Syrphe","Chenille",
          "Autre","Dyptere","Leopidoptere","Appis_melifera","Bourdon"        
          ]


vals, counts = np.unique(y_test, return_counts = True)
for val, count in zip(vals, counts):
    print(f"Label {val} has {count} occurences in y_test.")





def resample(X, occ):
    # Takes as argument a data X of length n
    # Will return occ random points of X
    N = len(X)
    return np.array([X[np.random.randint(0, N)] for _ in range(occ)])


X_train_resampled = []
y_train_resampled  = []
occurences = 1100

for label in range(11): # 11 labels
    mask = y_train == label

    X_temp = X_train[mask]
    y_temp = y_train[mask]

    for point in resample(X_temp, occurences):
        X_train_resampled.append(point)

    for _ in range(occurences):
        y_train_resampled.append(label)
    

X_train_resampled = np.array(X_train_resampled)
y_train_resampled = np.array(y_train_resampled).astype(int)

print("--"*20)
print(f"Shape  of resampled_X : {X_train_resampled.shape}")
print(f"Shape  of resampled_y : {y_train_resampled.shape}")
print("--"*20)

vals, counts = np.unique(y_train_resampled, return_counts=True)
for val, count in zip(vals, counts):
    print(f"Label {labels[val]} has {count} occurences in y_train.")

# Saving the files
'''
np.savez("data",
         X_train = X_train,
         X_test = X_test,
         y_train = y_train
         )


train_data = {
    "X" : X_train_resampled,
    "Y" : y_train_resampled
}'''


with open("y_test.json", "w") as f:
    json.dump(data, f)

np.save(os.path.join(os.path.dirname(__file__), "X_train.npy"), X_train_resampled)
np.save(os.path.join(os.path.dirname(__file__), "y_train.npy"), y_train_resampled)

# np.save("y_test.npy", y_test)





