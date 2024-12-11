import os
import pickle
import pandas as pd
import umap
import plotly.express as px
import numpy as np
import numpy
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

    speaker_path = os.path.join(base_directory, speaker_no, train_type)

    for label in os.listdir(speaker_path):
        print(label)
        feature,labelss = label.split("_")
        if feature == feature_type:
            label_feature = feature+"_"+labelss
            feature_path = os.path.join(speaker_path, label_feature)
            print(feature_path)

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


def visualize_umap(data, labels, feature_type, output_directory,model_no):
    """
    Apply UMAP and save the plots using Plotly.

    Parameters:
        data (DataFrame): Data with features.
        labels (List): Corresponding labels for the data.
        feature_type (str): The feature type being visualized (for file naming).
        output_directory (str): Directory to save the plots.
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
        title=f'UMAP Visualization of {feature_type}',
        labels={'label': 'Label'}
    )

    # Save the plot as HTML
    # os.makedirs(output_directory, exist_ok=True)
    # html_output_path = os.path.join(output_directory, f'{feature_type}_umap.html')
    # fig.write_html(html_output_path)
    # print(f"Plot saved to {html_output_path}")

    # Save the plot as PNG
    png_output_path = os.path.join(output_directory, f'{model_no}_{feature_type}_umap.png')
    fig.write_image(png_output_path)
    print(f"Plot saved to {png_output_path}")


# Main execution
if __name__ == "__main__":
    base_directory = "/data/FF_V2/Features"
    feature_types = ["w2vxls","wavlmlarge","unispeech"
  ]  # List of feature types
    speaker_no = "Donald_Trump_v2"
    train_type = "train"
    output_directory = f"/data/FF_V2/visualization/{speaker_no}"  # Directory to save plots
    import os
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    i=0
    for feature_type in feature_types:
        print(f"Processing feature type: {feature_type}")

        data_df = read_pickle_files(base_directory, feature_type, speaker_no, train_type)

        if not data_df.empty:
            visualize_umap(data_df, data_df['label'], feature_type, output_directory,str(i).zfill(3))
            i=i+1
        else:
            print(f"No data found for feature type: {feature_type}")
