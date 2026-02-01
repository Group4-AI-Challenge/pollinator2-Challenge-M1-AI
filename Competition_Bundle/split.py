import numpy as np




#X  = np.load("data_2/X.npy")
#y  = np.load("data_2/y.npy")
#angles = np.load("data_2/angles.npy")

#labels = np.unique(y)



def get_angles(label, m):
    # Returns the angles where a label appears
    line = m[label]
    return np.array([i for i in range(len(line)) if line[i]])


def split_label(label, matrix, test_size, random_state):
    # R


    angles_temp = get_angles(label, matrix)
    np.random.shuffle(angles_temp)
    angles_count = len(angles_temp)
    
    angleTestSize = max(1, int(test_size * angles_count))
    angleTrainSize = angles_count - angleTestSize

    trainAngles = angles_temp[:angleTrainSize]
    testAngles = angles_temp[angleTrainSize:]
    
    trainMask = (y == label) & (np.isin(angles, trainAngles))
    testMask = (y == label) & (np.isin(angles, testAngles))

    return trainMask, testMask


def split_labels(labels, N, matrix, test_size, random_state):
    train_mask = np.array([False for _ in range(N)])
    testMask = np.array([False for _ in range(N)])
    
    for label in labels:
        train, test = split_label(label, matrix, test_size)
        train_mask = train_mask | train
        testMask = testMask | test

    return train_mask, testMask



def split_data(X, y, angles, random_state = 42, return_mask = False):
    '''
    Docstring for split_data
        Split X and y into X_train, X_test, y_train and y_test so that
        the angles for one given specie is different in the train and in the test.


    :param X: An array containing the data points
    :param y: An array containing the labels of the points
    :param angles : An array containing the angles of each point
    :param random_state: The state for selecting the angles
    :param return_mask: A boolean set to False by default. If set to t

    Return : X_train, X_test, y_train, y_test if return mask is set to False
        a mask for the train and one for the test
        
    '''



    unique_classes = np.sort(np.unique(y))
    matrix =  np.zeros((11, 31), dtype=int)

    for label, angle in zip(y, angles):
        matrix[label][angle] = 1

    anglesLabel = np.sum(matrix , axis = 1)

    labels_kept = np.array([i for i, count in enumerate(anglesLabel) if count > 1])

    mask = np.isin(y, labels_kept)

    X = X[mask]
    y = y[mask]
    angles = angles[mask]

    train_mask, test_mask = split_labels(labels_kept, len(X), random_state)

    if return_mask:
        return train_mask, test_mask
    else:
        return X[train_mask], X[test_mask], y[train_mask], y[test_mask]



    


'''


X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
angles_train = np.load("angles.npy")


vals, counts = np.unique(y_train, return_counts=True)

for val, c in zip(vals, counts):
    print(f"Label {val} appeared {c} times.")

'''

    






