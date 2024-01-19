from poisoned_data.indices_poisoned_data import load_indices_poisoned_data
from features.extract_features import load_extracted_features
from features.seperate_poisoned import separate
from data_filtering.kNN_filter import kNN

indices_of_poisoned = load_indices_poisoned_data()
extracted_features_X, extracted_features_Y = load_extracted_features() #load the extracted features from the file, of shape 50000 x 512 and the labels of features of shape 50000 x 1

clean_data, bad_data = [], []
lamda = 80 #threshold of relabeling

features_poisoned, labels_poisoned, features_clean, labels_clean = separate(extracted_features_X, indices_of_poisoned, extracted_features_Y) #separating poisoned and clean features and labels

""" for i in range(10):
    print(f"Number of labels {str(i)} in labels of poisoned: {str(labels_poisoned.count(i))}")
    print(f"Number of labels {str(i)} in labels of clean: {str(labels_clean.count(i))}")
    print(f"Number of labels {str(i)} in labels total: {str(list(extracted_features_Y).count(i))}") """

pred_labels, label_confidence = kNN(extracted_features_X, extracted_features_Y)
# print(pred_labels)
# print(label_confidence)
counter = sum(lab1 == lab2 for lab1, lab2 in zip(pred_labels, extracted_features_Y))
print(f"Broj točno klasificiranih primjera prema danim oznakama: {str(counter)} i to je {str(counter * 100 / len(extracted_features_Y))}%")