def separate_data(features, indices, labels):
    features_poisoned, features_clean = [], []
    labels_poisoned, labels_clean = [], []
    for i in range(len(features)):
        if i in indices:
            features_poisoned.append(features[i])
            labels_poisoned.append(labels[i])
        else:
            features_clean.append(features[i])
            labels_clean.append(labels[i])

    return features_poisoned, labels_poisoned, features_clean, labels_clean

def separate_filtered_data(features, predicted_labels, labels):
    clean, bad = {}, {}
    for i, feat in enumerate(features):
        if predicted_labels[i] == labels[i]:
            clean[feat] = labels[i]
        else:
            bad[feat] = labels[i]

    return clean, bad
