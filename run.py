import json
import os
from pathlib import Path

from confusion_matrix import show_confusion_matrix
from data_filtering.kNN_filter import KNN
from features.extract_features import load_extracted_features
from features.seperate_poisoned import separate_data
from poisoned_data.indices_poisoned_data import load_indices_poisoned_data
import test_BadNets 

SAVE_DIR_CONF = Path(__file__).parent / 'conf_matrices'

indices_of_poisoned = load_indices_poisoned_data()
test_BadNets.train_BadNets_noCleaning(indices_of_poisoned.tolist())
# extracted_features_X, extracted_features_Y = load_extracted_features() #load the extracted features from the file, of shape 50000 x 512 and the labels of features of shape 50000 x 1

# clean_data, bad_data = [], []
# lamda = 80 #threshold of relabeling

# features_poisoned, labels_poisoned, features_clean, labels_clean = separate_data(extracted_features_X, indices_of_poisoned, extracted_features_Y) #separating poisoned and clean features and labels

# for i in range(10):
#     print(f"Number of labels {str(i)} in labels of poisoned: {str(labels_poisoned.count(i))}")
#     print(f"Number of labels {str(i)} in labels of clean: {str(labels_clean.count(i))}")
#     print(f"Number of labels {str(i)} in labels total: {str(list(extracted_features_Y).count(i))}")

# classifier = KNN()

# if os.path.exists('pred_lab_conf_half.json'):
#     with open('pred_lab_conf_half.json', 'r') as file:
#         loaded_tuple = json.load(file)
#         pred_labels, label_confidence = loaded_tuple[0], loaded_tuple[1]
#     curr_dir = os.getcwd()
#     save_path = os.path.join(curr_dir, 'confusion_matrix_binar_half')
#     show_confusion_matrix(extracted_features_Y, pred_labels, indices_of_poisoned, save_path)
# else:
#     pred_labels, label_confidence = classifier.kNN_half(extracted_features_X, extracted_features_Y)
#     curr_dir = os.getcwd()
#     save_path = os.path.join(curr_dir, 'confusion_matrix_binar_half')
#     show_confusion_matrix(extracted_features_Y, pred_labels, indices_of_poisoned, save_path)
#     # Save tuple to a file
#     pred_lab_conf = (pred_labels.tolist(), label_confidence.tolist())
#     with open('pred_lab_conf_half.json', 'w') as file:
#         json.dump(pred_lab_conf, file)

# print(pred_labels)
# print(label_confidence)
# counter = sum(lab1 == lab2 for lab1, lab2 in zip(pred_labels, extracted_features_Y))
# print(f"Broj točno klasificiranih primjera prema danim oznakama: {str(counter)} i to je {str(counter * 100 / len(extracted_features_Y))}%")

# if os.path.exists('pred_lab_conf_10kNN.json'):
#     with open('pred_lab_conf_10kNN.json', 'r') as file:
#         loaded_tuple = json.load(file)
#         pred_labels, label_confidence = loaded_tuple[0], loaded_tuple[1]
#     save_path = os.path.join(SAVE_DIR_CONF, 'confusion_matrix_binar_10kNN')
#     show_confusion_matrix(extracted_features_Y, pred_labels, indices_of_poisoned, save_path)
# else:
#     pred_labels, label_confidence = classifier.kNN_10(extracted_features_X, extracted_features_Y)
#     save_path = os.path.join(SAVE_DIR_CONF, 'confusion_matrix_binar_10kNN')
#     show_confusion_matrix(extracted_features_Y, pred_labels, indices_of_poisoned, save_path)
#     counter = sum(lab1 == lab2 for lab1, lab2 in zip(pred_labels, extracted_features_Y))
#     print(f"Broj točno klasificiranih primjera prema danim oznakama: {str(counter)} i to je {str(counter * 100 / len(extracted_features_Y))}%")
#     # Save tuple to a file
#     pred_lab_conf = (pred_labels.tolist(), label_confidence)
#     with open('pred_lab_conf_10kNN.json', 'w') as file:
#         json.dump(pred_lab_conf, file)

# print(pred_labels)
# print(label_confidence)
