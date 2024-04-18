from collections import Counter

from sklearn.neighbors import KNeighborsClassifier


class KNN:

    def kNN_10(self, features, labels):
        counts = Counter(labels)
        dict_kNeighfitted = {}
        for label in counts.keys():
            count = counts.get(label)
            k_i = round(count / 2)
            kNeig_i = KNeighborsClassifier(n_neighbors=k_i, algorithm='brute').fit(features, labels)
            dict_kNeighfitted[label] = kNeig_i

        pred_labels = []
        label_confidences = []
        for i, sample in enumerate(features):
            kNeig = dict_kNeighfitted.get(labels[i])
            predicted_label = kNeig.predict(sample.reshape(1, -1))
            label_conf = kNeig.predict_proba(sample.reshape(1, -1))[0, predicted_label]
            pred_labels.append(predicted_label[0])
            label_confidences.append(label_conf[0])
            print(f"Sample with index {str(i)} has predicted label: {str(pred_labels[-1])} with confidence: {str(label_confidences[-1])} and true label of {str(labels[i])}")
        
        return pred_labels, label_confidences
    
    def kNN_half(self, features, labels):
        # counts = Counter(labels)
        # dict_kNeighfitted = {}
        # for label in counts.keys():
            # count = counts.get(label)
        n = round(len(features) / 2)
        kNeig = KNeighborsClassifier(n_neighbors=n, algorithm='brute').fit(features, labels)
            # dict_kNeighfitted[label] = kNeig_i

        pred_labels = []
        label_confidence = []
        for i, sample in enumerate(features):
            # kNeig = dict_kNeighfitted.get(labels[i])
            predicted_label = kNeig.predict(sample.reshape(1, -1))
            label_conf = kNeig.predict_proba(sample.reshape(1, -1))[0, predicted_label]
            pred_labels.append(predicted_label[0])
            label_confidence.append(label_conf[0])
            print(f"Sample with index {str(i)} has predicted label: {str(pred_labels[-1])} with confidence: {str(label_confidence[-1])} and true label of {str(labels[i])}")
        
        return pred_labels, label_confidence
