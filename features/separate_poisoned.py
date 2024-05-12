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

def filter_and_relabel_data(features, true_labels, predicted_labels, label_confidences, threshold=80):
    clean, bad = [], []
    conf_to_check = []
    
    # FILTERING
    for i, feat in enumerate(features):
        if predicted_labels[i] == true_labels[i]:
            clean.append((feat, true_labels[i]))
        else:
            bad.append((feat, true_labels[i]))
            conf_to_check.append(i)

    print(f"Out of {len(true_labels)} total, {len(clean)} were filtered as clean and {len(bad)} as bad.")
    bef_relab_bad = len(bad)

    # RELABELING
    for i, conf_indx in zip(range(len(bad)), conf_to_check):
        if label_confidences[conf_indx] > threshold*1e-2:
            clean.append(bad[i])

    print(f"Out of {bef_relab_bad} filtered as bad, {len(bad)} remained bad and {bef_relab_bad - len(bad)} were added to clean.")
    return clean
