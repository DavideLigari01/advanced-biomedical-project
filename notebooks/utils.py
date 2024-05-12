# Utility functions for the notebooks

# ----------------- Import Libraries ---------------------------

import os
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
from tqdm.notebook import tqdm
import numpy as np
import random
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import spearmanr
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTEN
from collections import Counter

# ----------------- Functions ---------------------------


def move_corrupted_files(src_dir: str, dest_dir: str) -> tuple:
    """
    Checks for corrupted audio files in the source directory and moves them to a destination directory.

    Args:
        src_dir (str): The path to the source directory containing audio files.
        dest_dir (str): The path to the destination directory where corrupted files will be moved.

    Returns:
        tuple: A tuple containing the count of corrupted files found and a list of their filenames.
    """
    count = 0
    filenames = []

    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Iterate through files in the source directory
    for file in tqdm(os.listdir(src_dir), desc="Checking files"):
        try:
            # Attempt to load the audio file
            _ = torchaudio.load(os.path.join(src_dir, file))
        except:
            # Move the corrupted file to the destination directory
            os.system(
                f"mv {os.path.join(src_dir, file)} {os.path.join(dest_dir, file)}"
            )

            # Print an error message and increment the count of corrupted files
            print(ValueError(f"File {file} is corrupted"))
            count += 1
            filenames.append(file)

    # Print the total count of corrupted files found
    print(f"Files corrupted in {src_dir}:\t{count}\n")

    # Return the count of corrupted files and their filenames
    return count, filenames


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def slicing(
    waveform: np.ndarray | torch.Tensor, offset: int = 0, num_frames: int | None = None
) -> np.ndarray:
    """
    Slice the waveform into a specified number of frames starting from a given offset.

    Parameters:
    waveform (ndarray | tensor): The input waveform.
    offset (int): The starting offset for slicing the waveform. Default is 0.
    num_frames (int): The number of frames to slice from the waveform. Default is None, which slices until the end of the waveform.

    Returns:
    ndarray: The sliced waveform.

    """
    waveform = waveform[:, offset:num_frames]
    return waveform


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def extract_features(
    dir_path: str,
    label: str,
    frame_length: float,
    sample_rate: int = 44100,
    n_mfcc: int = 13,
    melkwargs: dict = {},
) -> list:
    """
    Extracts audio features (MFCC, chroma, RMS, spectral centroid, spectral bandwidth, spectral rolloff, zero-crossing rate)
    from audio files in the specified directory.

    Args:
        dir_path (str): The path to the directory containing audio files.
        label (str): The label associated with the extracted features.
        frame_length: The frame length for feature extraction.
        sample_rate (int, optional): The sample rate of the audio files. Defaults to 44100 Hz.
        n_mfcc (int, optional): The number of Mel-frequency cepstral coefficients (MFCCs) to extract. Defaults to 13.
        melkwargs (dict, optional): Additional arguments for Mel spectrogram computation.

    Returns:
        list: A list containing tensors of extracted features for each audio file.
    """
    filenames = os.listdir(dir_path)
    filenames = [file for file in filenames if not file.startswith(".")]
    frame_length = int(1 / frame_length)

    features = []  # List to store the features for all files

    for file in tqdm(filenames, desc=f"Extraction in progress"):
        audio, orig_sample_rate = torchaudio.load(os.path.join(dir_path, file))
        orig_sample_rate_tmp = int(orig_sample_rate / frame_length)
        audio_length = int(max(audio[0].shape) / (orig_sample_rate_tmp))

        sample_rate = orig_sample_rate

        # Reduce the audio from stereo to mono if needed
        if audio.shape[0] > 1:
            print(f"Converting stereo audio to mono for {file}...")
            audio = torch.mean(audio, dim=0).reshape(1, -1)

        features_local = []  # List to store the MFCC features for each file

        for i in range(audio_length):
            # Lazy load the audio, one sec at a time, to avoid memory issues
            audio_mono = slicing(
                audio,
                offset=int(sample_rate / frame_length * i),
                num_frames=int(sample_rate / frame_length * (i + 1)),
            )

            # Get the MFCC features
            mfcc_features_tmp = extract_mfcc(audio_mono, sample_rate, n_mfcc, melkwargs)
            chroma_stft = extract_chroma_stft(audio_mono, sample_rate)
            rms = extract_rms(audio_mono)
            spec_cent = extract_spectral_centroid(audio_mono, sample_rate)
            spec_bw = extract_spectral_bandwidth(audio_mono, sample_rate)
            rolloff = extract_spectral_rolloff(audio_mono, sample_rate)
            zcr = extract_zero_crossing_rate(audio_mono)

            # Concatenate the features
            features_tmp = torch.cat(
                (mfcc_features_tmp, chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr),
                dim=0,
            )
            features_local.append(features_tmp)

        # If features_local is empty, skip the file
        if features_local == []:
            print(f"No features extracted for {file}. Skipping...")
            continue

        # Stack the MFCC features into a single tensor
        features_local = torch.stack(features_local, dim=0)

        # Attach the label in the last column
        features_local = torch.cat(
            (features_local, torch.ones(features_local.shape[0], 1) * label), dim=1
        )

        # Attach the filename index in the last column
        features_local = torch.cat(
            (
                features_local,
                torch.full((features_local.shape[0], 1), filenames.index(file)),
            ),
            dim=1,
        )

        features.append(features_local)

    print("Finished processing all files.\n")
    return features


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def save_features(data: torch.Tensor, path: str, name: str) -> None:
    """
    Save the features to a file.

    Args:
        data (torch.Tensor): The data to be saved.
        path (str): The path to save the data.
        name (str): The name of the file.

    Returns:
        None
    """

    # create the directory if it does not exist
    os.makedirs(path, exist_ok=True)

    # if features already exixts, inform the user
    if os.path.exists(path + name):
        print(
            "Features already exist! Please check the directory to avoid overwriting the data."
        )
        return None

    else:
        print("Saving features...")
        # save the data
        np.savez_compressed(
            path + name, X=data[:, :-2], y=data[:, -2], filename=data[:, -1]
        )

    return None


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def extract_mfcc(
    audio: torch.Tensor, sample_rate: int, n_mfcc: int, melkwargs: dict
) -> torch.Tensor:
    """
    Extract MFCC features from audio files. It takes the directory path containing the audio files and returns a list of tensors containing MFCC features for each audio file.
    Each element in the list is a tensor of shape (num_frames, n_mfcc+2), where num_frames is the number of frames in the audio file, n_mfcc is the number of MFCC coefficients,
    and the last two columns are the label and the filename.
    The MFCCs are computed on a 1 sec audio segment at a time. As a consequence an audio file with a length of 10 sec will have 10 tensors in the list. Note that if
    the audio file is stereo, it will be converted to mono before computing the MFCCs.

    Args:
    - audio (torch.Tensor): The audio file.
    - sample_rate (int): The sample rate of the audio file.
    - n_mfcc (int): The number of MFCC coefficients to compute.
    - melkwargs (dict): Additional arguments for Mel-scale transformation.

    Returns:
    - mfcc_features (torch.Tensor): The MFCC features of the audio file.
    """
    # Create the MFCC transform
    mfcc_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs=melkwargs)
    mfcc_features = mfcc_transform(audio)
    return torch.mean(mfcc_features, dim=2).reshape(
        -1
    )  # Take the mean of the MFCC coefficients over time. The number of columns depends on the hop_length.
    # return mfcc_features


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def extract_chroma_stft(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Extract Chroma STFT features from an audio file.

    Args:
    - audio (torch.Tensor): The audio file.
    - sample_rate (int): The sample rate of the audio file.

    Returns:
    - chroma_stft (torch.Tensor): The Chroma STFT features of the audio file.
    """
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio.numpy(), sr=sample_rate))
    chroma_stft = torch.tensor(chroma_stft)
    return chroma_stft.unsqueeze(0)


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def extract_rms(audio: torch.Tensor) -> torch.Tensor:
    """
    Extract RMS features from an audio file.

    Args:
    - audio (torch.Tensor): The audio file.

    Returns:
    - rms (torch.Tensor): The RMS features of the audio file.
    """
    rms = np.mean(librosa.feature.rms(y=audio.numpy()))
    rms = torch.tensor(rms)
    return rms.unsqueeze(0)


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def extract_spectral_centroid(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Extract Spectral Centroid features from an audio file.

    Args:
    - audio (torch.Tensor): The audio file.
    - sample_rate (int): The sample rate of the audio file.

    Returns:
    - spec_cent (torch.Tensor): The Spectral Centroid features of the audio file.
    """
    spec_cent = np.mean(
        librosa.feature.spectral_centroid(y=audio.numpy(), sr=sample_rate)
    )
    spec_cent = torch.tensor(spec_cent)
    return spec_cent.unsqueeze(0)


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def extract_spectral_bandwidth(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Extract Spectral Bandwidth features from an audio file.

    Args:
    - audio (torch.Tensor): The audio file.
    - sample_rate (int): The sample rate of the audio file.

    Returns:
    - spec_bw (torch.Tensor): The Spectral Bandwidth features of the audio file.
    """
    spec_bw = np.mean(
        librosa.feature.spectral_bandwidth(y=audio.numpy(), sr=sample_rate)
    )
    spec_bw = torch.tensor(spec_bw)
    return spec_bw.unsqueeze(0)


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def extract_spectral_rolloff(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Extract Spectral Rolloff features from an audio file.

    Args:
    - audio (torch.Tensor): The audio file.
    - sample_rate (int): The sample rate of the audio file.

    Returns:
    - rolloff (torch.Tensor): The Spectral Rolloff features of the audio file.
    """
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio.numpy(), sr=sample_rate))
    rolloff = torch.tensor(rolloff)
    return rolloff.unsqueeze(0)


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def extract_zero_crossing_rate(audio: torch.Tensor) -> torch.Tensor:
    """
    Extract Zero Crossing Rate features from an audio file.

    Args:
    - audio (torch.Tensor): The audio file.

    Returns:
    - zcr (torch.Tensor): The Zero Crossing Rate features of the audio file.
    """
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio.numpy()))
    zcr = torch.tensor(zcr)
    return zcr.unsqueeze(0)


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def generate_samples(
    dir: str,
    target_num: int,
    noise_factor: list,
    speed_factor: list,
    pitch_factor: list,
    marker: str = "",
    random_seed: int = 42,
) -> None:
    """
    Generate additional audio samples to balance the dataset.

    This function generates additional audio samples by applying noise, speed, and pitch modifications to the existing audio files in a directory.

    Args:
            dir (str): The directory path where the audio files are located.
            target_num (int): The desired number of audio samples to generate.
            noise_factor (list): A list representing the range of noise factors to apply to the audio files.
            speed_factor (list): A list representing the range of speed factors to apply to the audio files.
            pitch_factor (list): A list representing the range of pitch factors to apply to the audio files.
            marker (str, optional): A marker to append to the generated file names. Defaults to ''.

    Returns:
            None
    """
    files = os.listdir(dir)
    num_files = len(files)
    to_generate = target_num - num_files

    if to_generate <= 0:
        print(f"No need to generate samples for {dir} directory")
        return
    else:
        percent = to_generate / num_files * 100
        # warn the user about the percentage of the data to be generated
        print(f"Generating {to_generate} samples ({percent:.2f}%) for {dir} directory")

    to_generate_noise = to_generate // 3
    to_generate_speed = to_generate // 3
    to_generate_pitch = to_generate - to_generate_noise - to_generate_speed

    try:
        files_noisy = random.sample(
            files, k=to_generate_noise
        )  # sample without replacement
    except:
        files_noisy = random.choices(
            files, k=to_generate_noise
        )  # sample with replacement
        # multiple names can be the same. This may lead to overwriting the files thus not reaching the target number of files.

    try:
        files_speed = random.sample(files, k=to_generate_speed)
    except:
        files_speed = random.choices(files, k=to_generate_speed)
    try:
        files_pitch = random.sample(files, k=to_generate_pitch)
    except:
        files_pitch = random.choices(files, k=to_generate_pitch)

    print("Generating noisy audio samples")
    for i, file in enumerate(files_noisy):
        noise_factor_ = random.uniform(noise_factor[0], noise_factor[1])
        audio, sr = torchaudio.load(dir + file)
        audio = audio.mean(0).reshape(1, -1).numpy()[0]
        noise = np.random.randn(len(audio)) * noise_factor_
        noisy_audio = audio + noise
        torchaudio.save(
            dir + f"noisy_{i}" + marker + file,
            torch.tensor(noisy_audio).unsqueeze(0),
            sr,
        )
        # the {i} solves the issue of overwriting the files thus maintaining the desired number of files

    print("Generating speed audio samples")
    for i, file in enumerate(files_speed):
        speed_factor_ = random.uniform(speed_factor[0], speed_factor[1])
        audio, sr = torchaudio.load(dir + file)
        audio = audio.mean(0).reshape(1, -1).numpy()[0]
        audio_speed = librosa.effects.time_stretch(audio, rate=speed_factor_)
        torchaudio.save(
            dir + f"speed_{i}" + marker + file,
            torch.tensor(audio_speed).unsqueeze(0),
            sr,
        )

    print("Generating pitch audio samples")
    for i, file in enumerate(files_pitch):
        pitch_factor_ = random.uniform(pitch_factor[0], pitch_factor[1])
        audio, sr = torchaudio.load(dir + file)
        audio = audio.mean(0).reshape(1, -1).numpy()[0]
        audio_pitch = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_factor_)
        torchaudio.save(
            dir + f"pitch_{i}" + marker + file,
            torch.tensor(audio_pitch).unsqueeze(0),
            sr,
        )

    print("Done!")


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def remove_generated_samples(dir: str, marker: str) -> None:
    """
    Remove the generated audio samples from the directory.

    This function removes the audio samples that were generated using the generate_samples function from the directory.

    Args:
            dir (str): The directory path where the audio files are located.
            marker (str): The marker used to identify the generated files.

    Returns:
            None
    """
    files = os.listdir(dir)
    generated_files = [file for file in files if marker in file]

    for file in generated_files:
        os.remove(dir + file)

    print(f"Removed {len(generated_files)} generated samples from {dir} directory")


# --------------------------------------------------------------------
# --------------------------------------------------------------------

import os
import shutil


def get_move_outliers(
    src_dir: str,
    out_dir: str,
    classname: str,
    outliers: pd.DataFrame,
    audio_info: pd.DataFrame,
    move: bool = False,
) -> list:
    """
    Retrieves and optionally moves outliers for a given class from the source directory to an output directory.

    Args:
        src_dir (str): The path to the source directory containing audio files.
        out_dir (str): The path to the output directory where outliers will be moved.
        classname (str): The name of the class for which outliers are to be handled.
        outliers: Dataframe containing information about outliers.
        audio_info: Dataframe containing information about all audio files.
        move (bool, optional): If True, move the outliers to the output directory. Defaults to False.

    Returns:
        list: A list of filenames of the outliers for the specified class.
    """
    # Retrieve outliers for the specified class
    out = outliers[outliers["label"] == classname]["filename"].to_list()
    print(f"{classname} outliers {out}")
    print(f"number of {classname} outliers {len(out)}")

    # Drop the outliers for the specified class
    audio_info_outliers = audio_info[~audio_info["filename"].isin(out)]

    print("directory size", len(os.listdir(src_dir)))

    if not move:
        return out
    elif move:
        # Move the outliers to the output directory
        os.makedirs(os.path.join(out_dir, classname), exist_ok=True)
        for file in out:
            shutil.move(os.path.join(src_dir, file), os.path.join(out_dir, classname))

        print("post-removal directory size", len(os.listdir(src_dir)))
        return out
    else:
        print("Invalid move value. Please enter True or False")
        return None


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def print_correlation(
    data_df: pd.DataFrame, title: str, pvalue: bool = True, figuresize: tuple = (20, 9)
) -> tuple:
    """
    Visualizes Spearman correlation matrix and corresponding p-values for a given DataFrame.
    This function calculates Spearman correlation coefficients and p-values for all pairs of columns
    in the DataFrame and visualizes them using heatmaps. If pvalue is set to True, it displays both
    the correlation matrix and the corresponding p-values. The size of the figure can be adjusted
    using the figuresize parameter.

    Parameters:
    - data_df (DataFrame): The DataFrame containing the data.
    - title (str): Title for the correlation plot.
    - pvalue (bool, optional): Whether to display p-values along with correlation values (default is True).
    - figuresize (tuple, optional): Size of the figure (default is (20, 9)).

    Returns:
    - correlation_matrix (DataFrame): DataFrame containing Spearman correlation coefficients.
    - p_values (DataFrame): DataFrame containing p-values corresponding to correlation coefficients.
    """
    # Set up figure and styling
    plt.figure(figsize=figuresize)
    if pvalue:
        sns.set_theme(context="paper", font_scale=1.4)
        plt.suptitle(title, fontsize=22, color="black")

    # Step 3: Calculate Spearman correlation coefficients and p-values
    correlation_matrix = pd.DataFrame(
        index=data_df.columns, columns=data_df.columns, dtype=float
    )
    p_values = pd.DataFrame(index=data_df.columns, columns=data_df.columns, dtype=float)

    # Calculate Spearman correlation coefficients and p-values for each column pair in the dataframe
    for col1 in data_df.columns:
        for col2 in data_df.columns:
            correlation, p_value = spearmanr(data_df[col1], data_df[col2])
            correlation_matrix.loc[col1, col2] = correlation
            p_values.loc[col1, col2] = p_value

    # Step 4: Visualize the correlation matrix

    # Visualize the correlation matrix
    if pvalue:
        plt.subplot(1, 2, 1)
    sns.heatmap(correlation_matrix, annot=True, cmap="mako", linewidths=0.5)
    plt.yticks(rotation=0)
    plt.title("Spearman correlation", fontsize=16, color="black")

    if pvalue:
        # Visualize the p-values
        plt.subplot(1, 2, 2)
        sns.heatmap(p_values, annot=True, cmap="mako_r", fmt=".2f", linewidths=0.5)
        plt.yticks(rotation=0)
        plt.title("P-value", fontsize=16, color="black")

        plt.tight_layout()  # This will adjust subplots to fit into figure area.
    plt.show()
    return correlation_matrix, p_values


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def detect_outliers_iqr(
    data_df: pd.DataFrame, q1: float = 0.25, q3: float = 0.75
) -> pd.DataFrame:
    """
    Detects outliers in a DataFrame using the Interquartile Range (IQR) method.

    Parameters:
    - data_df (DataFrame): The input DataFrame.
    - q1 (float, optional): The percentile for the first quartile (default is 0.25).
    - q3 (float, optional): The percentile for the third quartile (default is 0.75).

    Returns:
    - outliers_df (DataFrame): DataFrame containing the outliers.

    This function calculates the first quartile (Q1), third quartile (Q3), and the interquartile range (IQR)
    for each column in the DataFrame. It then identifies outliers using the lower and upper bounds defined
    by Q1 - 2 * IQR and Q3 + 2 * IQR respectively. Outliers are detected using a boolean mask and
    filtered from the DataFrame.
    """
    Q1 = data_df.quantile(q1)
    Q3 = data_df.quantile(q3)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR

    # Detect outliers
    outliers_mask = (data_df < lower_bound) | (data_df > upper_bound)
    outliers_df = data_df[outliers_mask.any(axis=1)]

    return outliers_df


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def rebalance_data(data_to_balance: np.array, target_size: int, random_seed: int = 42) -> np.array:
    """
    Rebalances the data by oversampling or undersampling the classes based on the target size.
     
    Parameters:
    - data_to_balance: (np.array) representing the data to be rebalanced (X,y,filename).
    - target_size (int): The desired size for each class after rebalancing.
    - random_seed (int): The random seed for reproducibility.
    
    Returns:
    - data_rebalanced: Rebalanced data, where each data element is a numpy array.
    """
    
    random.seed(random_seed)
    
    # Extract the data and the labels
    X = data_to_balance[:,:-2]
    y = data_to_balance[:,-2]
    filenames = data_to_balance[:,-1]
    
    X = np.hstack((X, filenames.reshape(-1,1)))
    
    # Get the unique classes and their counts
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    # check the classes to be undersampled or oversampled
    classes_to_undersample = unique_classes[class_counts > target_size]
    classes_to_oversample = unique_classes[class_counts < target_size]
    
    # Initialize the undersampler and oversampler
    undersampler = RandomUnderSampler(sampling_strategy={class_: target_size for class_ in classes_to_undersample}, random_state=random_seed)
    oversampler = SMOTEN(sampling_strategy={class_: target_size for class_ in classes_to_oversample}, random_state=random_seed)
    
    # get the data to be undersampled
    X_to_undersample = X[np.isin(y, classes_to_undersample)]
    X_to_oversample = X[np.isin(y, classes_to_oversample)]
    
    # get the labels to be undersampled
    y_to_undersample = y[np.isin(y, classes_to_undersample)]
    y_to_oversample = y[np.isin(y, classes_to_oversample)]
    
    # undersample the data
    try:
        X_undersampled, y_undersampled = undersampler.fit_resample(X_to_undersample, y_to_undersample)
    except:  #when only one class is undersampled
        X_undersampled = np.array(random.sample(list(X_to_undersample), target_size))
        y_undersampled = np.array(random.sample(list(y_to_undersample), target_size))
        
    X_oversampled, y_oversampled = oversampler.fit_resample(X_to_oversample, y_to_oversample)
    
    # detach the filenames
    filenames_undersampled = X_undersampled[:,-1]
    filenames_oversampled = X_oversampled[:,-1]
    
    # remove the filenames from the data
    X_undersampled = X_undersampled[:,:-1]
    X_oversampled = X_oversampled[:,:-1]
    
    # concatenate the data
    data_oversampled = np.hstack((X_oversampled, y_oversampled.reshape(-1,1), filenames_oversampled.reshape(-1,1)))
    data_undersampled = np.hstack((X_undersampled, y_undersampled.reshape(-1,1), filenames_undersampled.reshape(-1,1)))
    full_data = np.vstack((data_oversampled, data_undersampled))
    
    return full_data
    
    
    
# --------------------------------------------------------------------
# --------------------------------------------------------------------