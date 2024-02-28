import numpy as np


def load_extracted_features(path='C:\\Users\\Luka\\GitHub\\Seminar\\features\\extracted_features.npz', allow_pickle=True):
    data = np.load(path)
    lst = data.files
    poisoned_10_trainX, poisoned_10_trainY = None, None
    for item in lst:
        if item == 'train_X':
            poisoned_10_trainX = data[item]
        elif item == 'train_y':
            poisoned_10_trainY = data[item]

    return poisoned_10_trainX, poisoned_10_trainY

def check_extracted_features():
    feat_poisoned_10_trainX, feat_poisoned_10_trainY = load_extracted_features()
    print("Učitani podaci su s veličinama:", feat_poisoned_10_trainX.shape, " i " , feat_poisoned_10_trainY.shape)
    print("Duljina značajki od otrovanog skupa za treniranje: " + str(len(feat_poisoned_10_trainX)))
    print("Duljina oznaka od otrovanog skupa za treniranje: " + str(len(feat_poisoned_10_trainY)))
    element = feat_poisoned_10_trainX[0]
    print("Prvi element značajki od otrovanog skupa za treniranje: " + str(element))
    mark = feat_poisoned_10_trainY[0]
    print("Oznaka prvi elementa od otrovanog skupa za treniranje: " + str(mark))

# check_extracted_features()