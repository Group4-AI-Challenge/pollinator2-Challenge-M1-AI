import numpy as np



 
def get_angles(label, m):
    '''
    Calculates the angles where a label appears
    
    :param label: An integer representing the label
    :param m: The matrix containing the angles information
    Return : An array containing the angles where the label appears

    ## This function is not supposed to be used directly. Use split_data instead.
    '''
    line = m[label]
    return np.array([i for i in range(len(line)) if line[i]])


def split_label(X, y, angles,label, matrix, test_size):
    '''
    Splits the data for a given label into train and test masks
    
    :param X: An array containing the data points
    :param y: An array containing the labels of the points
    :param angles: An array containing the angles of each point
    :param label: An integer representing the label to be split
    :param matrix: The matrix containing the angles information
    :param test_size: A float between 0 and 1 representing the proportion of data to be in the test set
    :param random_state: The random state for shuffling the angles
    Return : The trainMask, testMask for the label

    ## This function is not supposed to be used directly. Use split_data instead.
    '''


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


def split_labels(X, y,angles,labels, N, matrix, test_size):
    '''
    Splits the data for all labels into train and test masks
    
    :param X: An array containing the data points
    :param y: An array containing the labels of the points
    :param angles: An array containing the angles of each point
    :param labels: An array containing the labels to be split
    :param N: The number of data points
    :param matrix: The matrix containing the angles information
    :param test_size: A float between 0 and 1 representing the proportion of angles to be in the test set
    :param random_state: The random state for shuffling the angles

    ## This function is not supposed to be used directly. Use split_data instead.
    '''
    train_mask = np.array([False for _ in range(N)])
    testMask = np.array([False for _ in range(N)])
    
    for label in labels:
        train, test = split_label(X, y, angles, label, matrix, test_size)
        train_mask = train_mask | train
        testMask = testMask | test

    return train_mask, testMask



def split_data(X, y, angles, test_size = 0.7, return_mask = False):
    '''
    Docstring for split_data
        Split X and y into X_train, X_test, y_train and y_test so that
        the angles for one given specie is different in the train and in the test.

    :param X: An array containing the data points
    :param y: An array containing the labels of the points
    :param angles : An array containing the angles of each point
    :param random_state: The state for selecting the angles
    :param return_mask: A boolean set to False by default. 
    If set to to True, the function returns two masks instead of the splitted data.

    Return : X_train, X_test, y_train, y_test if return mask is set to False
        a mask for the train and one for the test
        
    ## This function is the main function to be used for splitting the data.
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

    train_mask, test_mask = split_labels(X, y, angles, labels_kept, len(X), matrix, test_size)
    if return_mask:
        return train_mask, test_mask
    else:
        return X[train_mask], X[test_mask], y[train_mask], y[test_mask], angles[train_mask], angles[test_mask]
    
