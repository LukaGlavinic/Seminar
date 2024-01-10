import numpy as np
import matplotlib.pyplot as plt

def load_extracted_features(path='C:\\Users\\Luka\\GitHub\\Seminar\\extracted_features\\extracted_features.npz', allow_pickle=True):
    data = np.load(path)
    lst = data.files
    #print('Duljina liste je:' + str(len(lst)))
    poisoned_10_trainX, poisoned_10_trainY = None, None
    for item in lst:
        print(item)
        if item == 'train_X':
            poisoned_10_trainX = data[item]
            image_index = 0
            reshaped = np.reshape(data[item][image_index], (16, 32))
            plt.imshow(reshaped)
            plt.title(f'CIFAR-10 Image at index {image_index}')
            plt.show()
        elif item == 'train_y':
            poisoned_10_trainY = data[item]
        #print('Oblik slike je:' + str(data[item][0].shape))
        #if item == 'train_y':
            #print(data[item])
        #print(len(data[item]))

    return poisoned_10_trainX, poisoned_10_trainY

load_extracted_features()