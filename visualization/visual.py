import os
import pickle
import pandas as pd
import umap
import plotly.express as px

def read_pickle_files(base_directory, feature_type, speaker_no, train_type):
    """
    Reads pickle files for a given feature type, speaker, and train type.

    Parameters:
        base_directory (str): Path to the base directory.
        feature_type (str): Feature type to process (e.g., 'wave2vec2').
        speaker_no (str): Speaker number.
        train_type (str): Train type (e.g., 'train', 'test').

    Returns:
        DataFrame: Combined data with features and labels.
    """
    data = []
    labels = []

    speaker_path = os.path.join(base_directory, 'DFADD', speaker_no, train_type)

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

    # Combine into a DataFrame
    return pd.DataFrame({'features': data, 'label': labels})

def visualize_umap(data, labels):
    """
    Apply UMAP and plot the results using Plotly.

    Parameters:
        data (DataFrame): Data with features.
        labels (List): Corresponding labels for the data.
    """
    # UMAP transformation
    umap_model = umap.UMAP(n_components=2, random_state=42)
    reduced_data = umap_model.fit_transform(list(data['features']))

    # Create a DataFrame for visualization
    visualization_df = pd.DataFrame({
        'UMAP1': reduced_data[:, 0],
        'UMAP2': reduced_data[:, 1],
        'label': labels
    })

    # Plot using Plotly
    fig = px.scatter(
        visualization_df, 
        x='UMAP1', 
        y='UMAP2', 
        color='label', 
        title='UMAP Visualization of Features',
        labels={'label': 'Label'}
    )

    fig.show()

# Main execution
if __name__ == "__main__":
    base_directory = "/data/Deep_Fake_Data/Raw_data/Features_superb"
    feature_type = "wavlablm_ms_40k"
    speaker_no = "p236"
    train_type = "train"

    data_df = read_pickle_files(base_directory, feature_type, speaker_no, train_type)

    if not data_df.empty:
        visualize_umap(data_df, data_df['label'])
    else:
        print("No data found for the specified inputs.")
