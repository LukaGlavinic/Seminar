from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

def kNN(features, labels):
    counts = Counter(labels)
    dict_kNeighfitted = {}
    for label in counts.keys():
        count = counts.get(label)
        k_i = round(count / 2)
        kNeig_i = KNeighborsClassifier(n_neighbors=k_i).fit(features, labels)
        dict_kNeighfitted[label] = kNeig_i
        # pred_i = kNeig_i.predict()

    pred_labels = []
    label_confidence = []
    for i, sample in enumerate(features):
        kNeig = dict_kNeighfitted.get(labels[i])
        predicted_label = kNeig.predict(sample.reshape(1, -1))
        label_conf = kNeig.predict_proba(sample.reshape(1, -1))[0, -1]
        pred_labels.append(predicted_label)
        label_confidence.append(label_conf)
        print(f"Sample with index {str(i)} has predicted label: {str(pred_labels[-1])} with confidence: {str(label_confidence[-1])} and true label of {str(labels[i])}")
    
    return pred_labels, label_confidence