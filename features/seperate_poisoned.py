def separate(features, indices, labels):
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