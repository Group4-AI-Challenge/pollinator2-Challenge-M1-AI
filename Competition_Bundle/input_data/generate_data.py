import numpy as np
import matplotlib.pyplot as plt
import json

# Load metadata (angles) and labels (classes)
metadata = np.load('trainValTest_N=2197469_prod2_metadata_labeledOnly.npz')['arr_0']
labels = np.load('trainValTest_N=2197469_prod2_nObs=4242_labels_labeledOnly.npz')['y']
X = np.load("X.npy")
# Extract angles from first row
angles = metadata[0]

# Load class mapper
with open('class_type_mapper.json', 'r') as f:
    name2id = json.load(f)
id2name = {v: k for k, v in name2id.items()}


N = 2000

idx_zero = np.where(labels == 0)[0]
idx_nonzero = np.where(labels != 0)[0]

# Step 2: Pick only N rows from y==0
np.random.shuffle(idx_zero)
idx_zero_selected = idx_zero[:N]

# Step 3: Combine with all rows where y!=0
final_idx = np.concatenate([idx_zero_selected, idx_nonzero])

# Step 4: Select rows from X and y
X = X[final_idx]
y = labels[final_idx]
angles = angles[final_idx]

vals, counts = np.unique(y, return_counts=True)

for v, c in zip(vals, counts):
    print(f"Label {v} appeared {c} times.")

print("--"*40)

unique_classes = np.sort(np.unique(labels))
matrix =  np.zeros((len(unique_classes), 31), dtype=int)

for label, angle in zip(y, angles):
    matrix[label][angle] = 1

anglesLabel = np.sum(matrix , axis = 1)

test_size = 0.3

for i, count in enumerate(anglesLabel):
    
    if count > 1:
        status = "Kept" 
        angleTest = max(1, int(test_size * count))
        angleTrain = count - angleTest
        print(f"Label {i} appeared in {count} angles -> Kept in train : {angleTrain} in test : {angleTest}")
    else:
        print(f"Label {i} appeared in {count} angles -> Dropped")

def get_angles(label, m):
    # Return the angles where a label appears
    line = m[label]
    return np.array([i for i in range(len(line)) if line[i]])

labels_kept = np.array([i for i, count in enumerate(anglesLabel) if count > 1])

mask = np.isin(y, labels_kept)

X = X[mask]
y = y[mask]
angles = angles[mask]


def split_label(label):
    # Creates a train-test split for a label
    angles_temp = get_angles(label, matrix)
    angles_count = len(angles_temp)
    
    angleTestSize = max(1, int(test_size * angles_count))
    angleTrainSize = angles_count - angleTestSize

    trainAngles = angles_temp[:angleTrainSize]
    testAngles = angles_temp[angleTrainSize:]

    trainMask = (y == label) & (np.isin(angles, trainAngles))
    testMask = (y == label) & (np.isin(angles, testAngles))

    return trainMask, testMask


def split_labels(labels, N):
    # Create the train-test split of all the labels (return a mask)
    train_mask = np.array([False for _ in range(N)])
    testMask = np.array([False for _ in range(N)])
    
    for label in labels:
        train, test = split_label(label)
        train_mask = train_mask | train
        testMask = testMask | test

    return train_mask, testMask





train_mask, test_mask = split_labels(labels_kept, len(X))


X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

angles_train = angles[train_mask]
angles_test = angles[test_mask]
print("--"*40)
data = {"y_test":y_test.tolist()}


with open("y_test.json", "w") as f:
    json.dump(data, f)

np.save("X_test.npy", X_test)


valsTrain, countTrain = np.unique(y_train, return_counts = True)

for val, count in zip(valsTrain, countTrain):
    print(f"Label {val} appeared {count} times in the training set.")
print("--"*40)

valsTest, countTest = np.unique(y_test, return_counts = True)

for val, count in zip(valsTest, countTest):
    print(f"Label {val} appeared {count} times in the testing set.")
print("--"*40)


def resample(X, occ):
    # Takes as argument a data X of length n
    # Will return occ random points of X
    N = len(X)
    return np.array([X[np.random.randint(0, N)] for _ in range(occ)])

X_train_resampled = []
y_train_resampled  = []
occurences = 800

for label in labels_kept: # 11 labels
    mask = y_train == label

    X_temp = X_train[mask]
    y_temp = y_train[mask]

    for point in resample(X_temp, occurences):
        X_train_resampled.append(point)

    for _ in range(occurences):
        y_train_resampled.append(label)
    

X_train_resampled = np.array(X_train_resampled)
y_train_resampled = np.array(y_train_resampled).astype(int)


valsTrain, countTrain = np.unique(y_train_resampled, return_counts = True)

for val, count in zip(valsTrain, countTrain):
    print(f"Label {val} appeared {count} times in the training set.")
print("--"*40)

np.save("X_train.npy", X_train_resampled)
np.save("y_train.npy", y_train_resampled)



