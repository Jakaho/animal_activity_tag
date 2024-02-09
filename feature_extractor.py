import numpy as np
import dataprocessing
from collections import Counter

import pandas as pd
from scipy.stats import kurtosis
from scipy.stats import skew


def get_majority_label(labels):
    label_counts = Counter(labels)
    majority_label = label_counts.most_common(1)[0][0]  # Get the most common label
    return majority_label

def extract_features(filepath, mode):
    #filepath = 'datasets/pos_A.csv'
    windows_ac = dataprocessing.import_and_downsample(filepath, False, mode)  #
    all_time_domain_signals = dataprocessing.process_window(windows_ac, None)

    features_df = pd.DataFrame()
    #RAW ACCELEROMETER DATA

    # Initialize lists for all features
    mean_acc_x, mean_acc_y, mean_acc_z = [], [], []
    std_acc_x, std_acc_y, std_acc_z = [], [], []
    max_acc_x, max_acc_y, max_acc_z = [], [], []
    min_acc_x, min_acc_y, min_acc_z = [], [], []
    Q5_x, Q5_y, Q5_z = [], [], []
    Q95_x, Q95_y, Q95_z = [], [], []
    average_intensity = []
    label = []

    # Process each window
    for window in windows_ac:
        # Calculate means
        mean_acc_x.append(window['ax'].mean())
        mean_acc_y.append(window['ay'].mean())
        mean_acc_z.append(window['az'].mean())

        # Calculate standard deviations
        std_acc_x.append(window['ax'].std())
        std_acc_y.append(window['ay'].std())
        std_acc_z.append(window['az'].std())

        # Calculate maximums
        max_acc_x.append(window['ax'].max())
        max_acc_y.append(window['ay'].max())
        max_acc_z.append(window['az'].max())

        # Calculate minimums
        min_acc_x.append(window['ax'].min())
        min_acc_y.append(window['ay'].min())
        min_acc_z.append(window['az'].min())

        # Calculate 5th percentiles
        Q5_x.append(np.percentile(window['ax'], 5))
        Q5_y.append(np.percentile(window['ay'], 5))
        Q5_z.append(np.percentile(window['az'], 5))

        # Calculate 95th percentiles
        Q95_x.append(np.percentile(window['ax'], 95))
        Q95_y.append(np.percentile(window['ay'], 95))
        Q95_z.append(np.percentile(window['az'], 95))

        # Calculate average intensity
        amag = np.sqrt(window['ax']**2 + window['ay']**2 + window['az']**2)
        average_intensity.append(np.mean(amag))
        label.append(get_majority_label(window['label'].tolist()))

    # Create a DataFrame from the features
    features_df = pd.DataFrame({
        'mean_acc_x': mean_acc_x,
        'mean_acc_y': mean_acc_y,
        'mean_acc_z': mean_acc_z,
        'std_acc_x': std_acc_x,
        'std_acc_y': std_acc_y,
        'std_acc_z': std_acc_z,
        'max_acc_x': max_acc_x,
        'max_acc_y': max_acc_y,
        'max_acc_z': max_acc_z,
        'min_acc_x': min_acc_x,
        'min_acc_y': min_acc_y,
        'min_acc_z': min_acc_z,
        'Q5_x': Q5_x,
        'Q5_y': Q5_y,
        'Q5_z': Q5_z,
        'Q95_x': Q95_x,
        'Q95_y': Q95_y,
        'Q95_z': Q95_z,
        'average_intensity': average_intensity,
        'label': label
    })

    # Set the index name if needed
    features_df.index.name = 'window_id'


    #AC-COMPONENTS (TIME DOMAIN)

    mean_ac_x, mean_ac_y, mean_ac_z = [], [], []
    std_ac_x, std_ac_y, std_ac_z = [], [], []
    max_ac_x, max_ac_y, max_ac_z = [], [], []
    Q5_acx, Q5_acy, Q5_acz = [], [], []
    Q95_acx, Q95_acy, Q95_acz = [], [], []
    kurt_acx, kurt_acy, kurt_acz = [], [], []
    skew_acx, skew_acy, skew_acz = [], [], []

    # Iterate over each window and calculate features
    for window in windows_ac:
        mean_ac_x.append(window['ac_ax'].mean())
        mean_ac_y.append(window['ac_ay'].mean())
        mean_ac_z.append(window['ac_az'].mean())
        
        std_ac_x.append(window['ac_ax'].std())
        std_ac_y.append(window['ac_ay'].std())
        std_ac_z.append(window['ac_az'].std())
        
        max_ac_x.append(window['ac_ax'].max())
        max_ac_y.append(window['ac_ay'].max())
        max_ac_z.append(window['ac_az'].max())

        Q5_acx.append(np.percentile(window['ac_ax'], 5))
        Q5_acy.append(np.percentile(window['ac_ay'], 5))
        Q5_acz.append(np.percentile(window['ac_az'], 5))

        Q95_acx.append(np.percentile(window['ac_ax'], 95))
        Q95_acy.append(np.percentile(window['ac_ay'], 95))
        Q95_acz.append(np.percentile(window['ac_az'], 95))
        
        kurt_acx.append(kurtosis(window['ac_ax'], fisher=False))
        kurt_acy.append(kurtosis(window['ac_ay'], fisher=False))
        kurt_acz.append(kurtosis(window['ac_az'], fisher=False))
        
        skew_acx.append(skew(window['ac_ax']))
        skew_acy.append(skew(window['ac_ay']))
        skew_acz.append(skew(window['ac_az']))

    # Assuming you have a DataFrame called features_df to store the features
    features_df['mean_ac_x'] = mean_ac_x
    features_df['mean_ac_y'] = mean_ac_y
    features_df['mean_ac_z'] = mean_ac_z

    features_df['std_ac_x'] = std_ac_x
    features_df['std_ac_y'] = std_ac_y
    features_df['std_ac_z'] = std_ac_z

    features_df['max_ac_x'] = max_ac_x
    features_df['max_ac_y'] = max_ac_y
    features_df['max_ac_z'] = max_ac_z

    features_df['Q5_ac_x'] = Q5_acx
    features_df['Q5_ac_y'] = Q5_acy
    features_df['Q5_ac_z'] = Q5_acz

    features_df['Q95_ac_x'] = Q95_acx
    features_df['Q95_ac_y'] = Q95_acy
    features_df['Q95_ac_z'] = Q95_acz

    features_df['kurt_acx'] = kurt_acx
    features_df['kurt_acy'] = kurt_acy
    features_df['kurt_acz'] = kurt_acz

    features_df['skew_acx'] = skew_acx
    features_df['skew_acy'] = skew_acy
    features_df['skew_acz'] = skew_acz


    # Initialize a dictionary to hold the features for each frequency band and axis
    # as well as for each window
    features_dict = {}

    # Iterate over each axis, window ID, and frequency band
    for axis, windows in all_time_domain_signals.items():
        for windowID, bands in windows.items():
            for frequency_band, signal in bands.items():
                magnitudes = np.abs(signal)
                
                # Calculate the features
                rms_value = np.sqrt(np.mean(np.square(magnitudes)))
                std_value = np.std(magnitudes)
                min_val = np.min(magnitudes)
                max_val = np.max(magnitudes)
                
                # Generate feature names
                rms_feature_name = f'rms_{axis}_band{frequency_band}'
                std_feature_name = f'std_{axis}_band{frequency_band}'
                min_feature_name = f'min_{axis}_band{frequency_band}'
                max_feature_name = f'max_{axis}_band{frequency_band}'

                # Initialize nested dictionaries if not already present
                if windowID not in features_dict:
                    features_dict[windowID] = {}
                
                # Store the features in the dictionary
                features_dict[windowID][rms_feature_name] = rms_value
                features_dict[windowID][std_feature_name] = std_value
                features_dict[windowID][min_feature_name] = min_val
                features_dict[windowID][max_feature_name] = max_val

    # Convert the dictionary of features into a DataFrame where each key is a window ID
    # and each value is another dictionary of features for that window
    spectral_df = pd.DataFrame.from_dict(features_dict, orient='index')

    # If your original DataFrame is indexed by 'window_id' and you want to merge
    # the features into this DataFrame, you can do so as follows:
    # Your existing DataFrame with window_id as index
    features_df = features_df.join(spectral_df)


    return features_df
