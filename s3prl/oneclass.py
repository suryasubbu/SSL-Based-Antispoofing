import os
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------------------------
# 1) Read Pickle Files (Same as your original function)
# -------------------------------------------------
def read_pickle_files(base_directory, feature_type, speaker_no, train_type):
    data = []
    labels = []

    speaker_path = os.path.join(base_directory, 'COMB_US', speaker_no, train_type)

    for label in os.listdir(speaker_path):
        label_path = os.path.join(speaker_path, label)
        feature_path = os.path.join(label_path, feature_type)

        if not os.path.exists(feature_path):
            continue

        for file in os.listdir(feature_path):
            if file.endswith('.pkl'):
                file_path = os.path.join(feature_path, file)

                with open(file_path, 'rb') as f:
                    feature_data = pickle.load(f)
                    data.append(feature_data)
                    labels.append(label)

    return pd.DataFrame({'features': data, 'label': labels})

# -------------------------------------------------
# 2) Train and Evaluate One-Class SVM
# -------------------------------------------------
def train_and_evaluate_one_class_svm(data_df):
    """
    data_df: pd.DataFrame with columns ["features", "label"].
    This function trains a One-Class SVM on 90% of 'Original' data,
    then tests on the remaining 10% Original + all other labels.
    Returns metrics as a dictionary.
    """

    # Separate out Original vs Other
    original_data = data_df[data_df["label"] == "Original"]["features"].tolist()
    other_data = data_df[data_df["label"] != "Original"]["features"].tolist()
    other_labels = data_df[data_df["label"] != "Original"]["label"].unique().tolist()  # not strictly needed for labeling, but can help in analysis
    print(other_labels)
    # If no Original data or too few samples, handle gracefully
    if len(original_data) < 2:
        print("Not enough 'Original' data for training a One-Class SVM.")
        return None

    # Convert features to numpy arrays
    original_data = np.array(original_data, dtype=object)
    other_data = np.array(other_data, dtype=object)
    
    # Train-test split on the Original data: 90% train, 10% test
    train_original, test_original = train_test_split(
        original_data, 
        test_size=0.1, 
        random_state=42
    )
    
    # Combine test_original with all other_data to form the test set
    # We'll label test_original as +1 (inlier) and other_data as -1 (outlier)
    X_test = np.concatenate([test_original, other_data], axis=0)
    y_test = np.concatenate([np.ones(len(test_original)), -1 * np.ones(len(other_data))], axis=0)

    # -------------------------------------------------
    # Scale features
    # -------------------------------------------------
    # You might want to fit the scaler only on "train_original" 
    # to simulate real one-class detection scenario.
    scaler = StandardScaler()
    train_original_scaled = scaler.fit_transform(train_original.tolist())
    X_test_scaled = scaler.transform(X_test.tolist())

    # -------------------------------------------------
    # Train One-Class SVM
    # -------------------------------------------------
    # nu ~ proportion of outliers you expect among the training data
    # kernel can be 'rbf', 'linear', etc. 
    clf = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
    clf.fit(train_original_scaled)

    # -------------------------------------------------
    # Predict
    # -------------------------------------------------
    # SVM returns +1 for inliers, -1 for outliers
    y_pred = clf.predict(X_test_scaled)

    # -------------------------------------------------
    # Evaluate
    # -------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["Outlier (-1)", "Inlier (+1)"], zero_division=0)

    results_dict = {
        "accuracy": acc,
    }
    return results_dict

# -------------------------------------------------
# 3) Main Execution
# -------------------------------------------------
if __name__ == "__main__":
    base_directory = "/data/Deep_Fake_Data/Features_no_padding"
    feature_types = ["ssast_patch_base", "espnet_hubert_base_iter0", "ast",
                     "wav2vec2_base_s2st_es_voxpopuli", "xlsr_53", "hubert_large_ll60k",
                     "wav2vec2_large_lv60_cv_swbd_fsh", "wavlm_large"]
    speaker_nos = ["p266"]  # example
    train_type = "train"

    # We will store all results in a list of dicts
    results_list = []

    for speaker_no in speaker_nos:
        print(f"\n--- Speaker: {speaker_no} ---")
        for feature_type in feature_types:
            print(f"Processing feature type: {feature_type}")
            data_df = read_pickle_files(base_directory, feature_type, speaker_no, train_type)

            if data_df.empty:
                print(f"No data found for feature type: {feature_type}")
                continue

            # Train One-Class SVM and Evaluate
            results = train_and_evaluate_one_class_svm(data_df)
            if results is not None:
                # Add info about feature & speaker to results
                results["feature_type"] = feature_type
                results["speaker"] = speaker_no
                results_list.append(results)
    
    # Convert results to DataFrame for easy saving
    if results_list:
        final_df = pd.DataFrame(results_list)
        results_output_path = os.path.join(base_directory, "OneClassSVM_results_COMB_US.csv")
        final_df.to_csv(results_output_path, index=False)
        print(f"\nAll results saved to: {results_output_path}")
    else:
        print("\nNo valid results to save.")
