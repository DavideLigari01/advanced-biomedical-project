# Utility functions for the notebooks

# ----------------- Import Libraries ---------------------------

import os
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import librosa.display
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
from imblearn.over_sampling import SMOTEN, SMOTE, ADASYN, RandomOverSampler
from collections import Counter
from typing import List, Tuple

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
    n_cqt: int = 84,
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
        n_cqt (int, optional): The number of constant-Q transform (CQT) bins to extract. Defaults to 84.

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
            cqt = extract_cqt(audio_mono, sample_rate, n_cqt)
            rms = extract_rms(audio_mono)
            spec_cent = extract_spectral_centroid(audio_mono, sample_rate)
            spec_bw = extract_spectral_bandwidth(audio_mono, sample_rate)
            rolloff = extract_spectral_rolloff(audio_mono, sample_rate)
            zcr = extract_zero_crossing_rate(audio_mono)

            # Concatenate the features
            features_tmp = torch.cat(
                (
                    mfcc_features_tmp,
                    chroma_stft,
                    cqt,
                    rms,
                    spec_cent,
                    spec_bw,
                    rolloff,
                    zcr,
                ),
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
    audio: torch.Tensor, sample_rate: int, n_mfcc: int, melkwargs: dict = {}
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
    # chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio.numpy(), sr=sample_rate))
    chroma_stft = np.mean(
        librosa.feature.chroma_stft(y=audio.numpy(), sr=sample_rate), axis=2
    )

    chroma_stft = torch.tensor(chroma_stft)
    return chroma_stft.reshape(-1)


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


def extract_cqt(audio: torch.Tensor, sample_rate: int, n_cqt: int = 84) -> torch.Tensor:
    """
    Extract Constant-Q Transform (CQT) features from an audio file.

    Args:
    - audio (torch.Tensor): The audio file.
    - sample_rate (int): The sample rate of the audio file.
    - n_cqt (int): The number of CQT bins to compute. Default is 84.

    Returns:
    - cqt (torch.Tensor): The CQT features of the audio file.
    """
    cqt = np.abs(
        np.mean(librosa.cqt(y=audio.numpy(), sr=sample_rate, n_bins=n_cqt), axis=2)
    )  # abs to deal with complex numbers
    cqt = torch.tensor(cqt)
    return cqt.reshape(-1)


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
    sns.set_theme(context="paper", font_scale=1.4)
    if pvalue:
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


def rebalance_data(
    data_to_balance: np.array, target_size: int, random_seed: int = 42
) -> np.array:
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
    X = data_to_balance[:, :-2]
    y = data_to_balance[:, -2]
    filenames = data_to_balance[:, -1]

    X = np.hstack((X, filenames.reshape(-1, 1)))

    # Get the unique classes and their counts
    unique_classes, class_counts = np.unique(y, return_counts=True)

    # check the classes to be undersampled or oversampled
    classes_to_undersample = unique_classes[class_counts > target_size]
    classes_to_oversample = unique_classes[class_counts < target_size]

    # Initialize the undersampler and oversampler
    undersampler = RandomUnderSampler(
        sampling_strategy={class_: target_size for class_ in classes_to_undersample},
        random_state=random_seed,
    )
    oversampler = RandomOverSampler(
        sampling_strategy={class_: target_size for class_ in classes_to_oversample},
        random_state=random_seed,
    )

    # get the data to be undersampled
    X_to_undersample = X[np.isin(y, classes_to_undersample)]
    X_to_oversample = X[np.isin(y, classes_to_oversample)]

    # get the labels to be undersampled
    y_to_undersample = y[np.isin(y, classes_to_undersample)]
    y_to_oversample = y[np.isin(y, classes_to_oversample)]

    # undersample the data
    if len(classes_to_undersample) >= 1:
        try:
            X_undersampled, y_undersampled = undersampler.fit_resample(
                X_to_undersample, y_to_undersample
            )
        except:  # when only one class is undersampled
            X_undersampled = np.array(
                random.sample(list(X_to_undersample), target_size)
            )
            y_undersampled = np.array(
                random.sample(list(y_to_undersample), target_size)
            )

        # detach the filenames
        X_to_oversample = np.concatenate((X_to_oversample, X_undersampled), axis=0)
        y_to_oversample = np.concatenate((y_to_oversample, y_undersampled), axis=0)

        filenames_undersampled = X_undersampled[:, -1]
        X_undersampled = X_undersampled[:, :-1]
        data_undersampled = np.hstack(
            (
                X_undersampled,
                y_undersampled.reshape(-1, 1),
                filenames_undersampled.reshape(-1, 1),
            )
        )

    if len(classes_to_oversample) >= 1:
        if len(classes_to_undersample) > 1:
            X_oversampled, y_oversampled = oversampler.fit_resample(
                X_to_oversample, y_to_oversample
            )
        else:
            oversampler = RandomOverSampler(
                sampling_strategy={
                    class_: target_size for class_ in classes_to_oversample
                },
                random_state=random_seed,
            )
            X_oversampled, y_oversampled = oversampler.fit_resample(
                X_to_oversample, y_to_oversample
            )

        X_oversampled = X_oversampled[np.isin(y_oversampled, classes_to_oversample)]
        y_oversampled = y_oversampled[np.isin(y_oversampled, classes_to_oversample)]

        filenames_oversampled = X_oversampled[:, -1]
        X_oversampled = X_oversampled[:, :-1]
        data_oversampled = np.hstack(
            (
                X_oversampled,
                y_oversampled.reshape(-1, 1),
                filenames_oversampled.reshape(-1, 1),
            )
        )

    # concatenate the data if they are both available
    if len(classes_to_undersample) >= 1 and len(classes_to_oversample) >= 1:
        full_data = np.vstack((data_oversampled, data_undersampled))
    elif len(classes_to_undersample) > 1:
        full_data = data_undersampled
    else:
        full_data = data_oversampled

    return full_data


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def create_dataset(
    full_data: list, labels_col: int = -2
) -> torch.utils.data.dataset.TensorDataset:
    """
    Create a dataset from the data of the real and fake audio files. Convert tensor to a dataset using the columns (:,:-idx_filenames) as data and the column (;, -idx_filenames) as the label.

    Args:
    - data (torch.Tensor): The input data tensor containing the data of the real and fake audio files.
    - idx_filenames (int): The index of the label column in the tensor. Default is 2.

    Returns:
    - data (torch.utils.data.dataset.TensorDataset): The dataset containing the data of the real and fake audio files.
    """

    datasets = []

    for data in full_data:
        data = torch.utils.data.TensorDataset(
            data[:, :labels_col].type(torch.float),
            data[:, labels_col].type(torch.LongTensor),
        )
        datasets.append(data)

    return datasets


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def get_dataloaders(
    full_dataset: list, batch_size: int, shuffling: list = [True, False, False]
) -> torch.utils.data.DataLoader:
    """
    Get the dataloaders for the training, validation, and test sets.

    Args:
    - full_dataset (list): A list containing the datasets for training, validation, and test.
    - batch_size (int): The batch size for the dataloaders.
    - shuffling (list, optional): A list of boolean values indicating whether to shuffle the data for each dataset.
        The length of the list should be equal to the number of datasets in full_dataset.
        Default is [True, False, False].

    Returns:
    - dataloaders (list): A list containing the training, validation, and test dataloaders.
    """

    shuffling = shuffling
    dataloaders = []

    for i, dataset in enumerate(full_dataset):
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffling[i]
        )
        dataloaders.append(dataloader)

    return dataloaders


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def get_device():
    """
    Return the device to be used for training
    """

    dev = (
        "cuda"
        if torch.cuda.is_available()
        # else "mps"
        # if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {dev} device")

    return torch.device(dev)


# --------------------------------------------------------------------
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def train_model(dataloader, model, loss_fn, optimizer, device, get_loss_acc=False):
    """Training loop.
    This function trains the model for one epoch, iterating over the dataloader batches and using the optimizer to update the model's weights.
    It also computes the loss and accuracy 5 times and appends the values to the lists loss_list and acc_list.
    ----------
    dataloader: torch.utils.data.DataLoader, training data
    model: torch.nn.Module, neural network
    loss_fn: loss function
    optimizer: torch.optim, optimizer
    device: torch.device, device to be used for training
    ----------
    Returns if get_loss_acc is True:
    loss_list: list, loss values
    acc_list: list, accuracy values

    The values are appended to the lists 5 times during the epoch
    ----------
    """

    # model = model.to(device)
    loss_list = []
    acc_list = []

    size = len(dataloader.dataset)  # number of samples
    num_batches = len(dataloader)  # number of batches

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print the progress 1 times during the epoch
        if batch % (num_batches // 1) == 0:
            loss_val, acc = 0, 0
            loss_val, current = loss.item(), (batch + 1) * len(X)
            acc = (pred.argmax(1) == y).type(torch.float).sum().item() / len(y) * 100

            if get_loss_acc:
                loss_list.append(loss_val)
                acc_list.append(acc)

            print(f"loss: {loss:>7f} - acc: {acc:2.2f}% [{current:>5d}/{size:>5d}]")

    if get_loss_acc:
        return loss_list, acc_list


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def test_model(dataloader, model, loss_fn, device, get_loss_acc=False):
    """Evaluation loop.
    This function evaluates the model on the evaluation data, iterating over the dataloader batches.
    It also computes the loss and accuracy 5 times and appends the values to the lists loss_list and acc_list.
    ----------
    dataloader: torch.utils.data.DataLoader, evaluation data
    model: torch.nn.Module, neural network
    loss_fn: loss function
    device: torch.device, device to be used for evaluation
    ----------
    Returns if get_loss_acc is True:
    loss_list: list, loss values
    acc_list: list, accuracy values

    The values are appended to the lists 5 times during the epoch
    ----------
    """

    # model = model.to(device)

    size = len(dataloader.dataset)  # number of samples
    num_batches = len(dataloader)  # number of batches
    batch_size = dataloader.batch_size

    # Set the model in evaluation mode
    model.eval()
    test_loss, acc = 0, 0
    loss_list = []
    acc_list = []

    # Disable gradient computation for efficiency
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # Compute the accuracy
            acc += (pred.argmax(1) == y).type(torch.float).sum().item()

            if get_loss_acc and batch % (num_batches // 1) == 0:
                loss_list.append(test_loss / (batch + 1))
                acc_list.append(acc * 100 / ((batch + 1) * batch_size))

    test_loss /= num_batches
    acc /= size
    print(f"Test Error: \n Accuracy: {(100*acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    if get_loss_acc:
        return loss_list, acc_list


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def fit_model(
    epochs,
    train_dl,
    test_dl,
    model,
    loss_func,
    optimizer,
    device,
    model_path_store,
    path_to_store=None,
    get_loss_acc=False,
    evaluation_epochs=10,
    checkpoint_epochs=5,
):
    """Training and evaluation loop.
    This function trains the model for the specified number of epochs and evaluates it on the test data.
    It also computes the loss and accuracy 5 times and appends the values to the lists train_loss_list, train_acc_list, test_loss_list, test_acc_list.
    ----------
    epochs: int, number of epochs
    train_dl: torch.utils.data.DataLoader, training data
    test_dl: torch.utils.data.DataLoader, test data
    model: torch.nn.Module, neural network
    loss_fn: loss function
    optimizer: torch.optim, optimizer
    device: torch.device, device to be used for training
    model_path_store: path to store the model with the best loss
    path_to_store: path to store the model and optimizer parameters. If None, the model is not saved.
    get_loss_acc: bool, return loss and accuracy values
    checkpoint_epochs: int, number of epochs between each checkpoint
    ----------
    Returns if get_loss_acc is True:
    train_loss_list: list, training loss values
    train_acc_list: list, training accuracy values
    test_loss_list: list, test loss values
    test_acc_list: list, test accuracy values
    ----------
    """

    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    train_loss_list, train_acc_list = [], []
    test_loss_list, test_acc_list = [], []
    best_loss = 10000000
    model = model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_model(
            train_dl, model, loss_func, optimizer, device, get_loss_acc=True
        )

        # Log the running loss averaged per batch
        # for both training and validation
        # writer.add_scalars('Training vs. Validation Loss',
        #           { 'Training' : train_loss, 'Validation' : train_loss },
        #           epoch + 1)
        # writer.flush()

        if epoch % evaluation_epochs == 0:
            print(f"Epoch {epoch+1}\n-------------------------------")
            # Train
            train_loss_list.extend(train_loss)
            train_acc_list.extend(train_acc)

            # Evaluation
            test_loss, test_acc = test_model(
                test_dl, model, loss_func, device, get_loss_acc=True
            )
            test_loss_list.extend(test_loss)
            test_acc_list.extend(test_acc)

            # Save the model with the best loss
            if test_loss[-1] < best_loss:
                best_loss = test_loss[-1]
                model_scripted = torch.jit.script(model)  # Export to TorchScript
                model_scripted.save(model_path_store + "_best_loss.pth")
                # torch.save(model.state_dict(), model_path_store + '_best_loss.pth')

        # Save parameters in case of interruption prior to completion
        if epoch % checkpoint_epochs == 0 and path_to_store:
            checkpoint_info = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
            }
            torch.save(checkpoint_info, path_to_store)

    print("Done!")

    return train_loss_list, train_acc_list, test_loss_list, test_acc_list


# --------------------------------------------------------------------
# --------------------------------------------------------------------
def resample_audio(sample: np.array, orig_freq: int, new_freq: int) -> np.array:
    """
    Resamples an audio sample from the original frequency to a new frequency.

    Args:
        sample (array-like): The audio sample to be resampled.
        orig_freq (int): The original frequency of the audio sample.
        new_freq (int): The new frequency to resample the audio sample to.

    Returns:
        array-like: The resampled audio sample.

    """
    resampler = T.Resample(orig_freq=orig_freq, new_freq=new_freq)
    sample_resampled = resampler(sample)
    return sample_resampled


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def check_resample_sample_rate(
    target_freq: int, src_dir: str, dest_dir: str = None, overwrite: bool = False
) -> None:
    """
    Check if all files in the source directory have the same sample rate.
    If not resample the audio to the target sample rate and save it in the destination directory if overwrite is set to False.
    Otherwise, overwrite the original file with the resampled audio.

    Parameters:
    - target_freq (int): The target sample rate to check against.
    - src_dir (str): The directory path where the audio files are located.
    - dest_dir (str): The directory path where the resampled audio files will be saved.
    - overwrite (bool, optional): Whether to overwrite the original file with the resampled audio.
      If set to False, the resampled audio will be saved with a new filename.

    Returns:
    None
    """

    # get filenames
    filenames = os.listdir(src_dir)
    # remove hidden files
    filenames = [file for file in filenames if not file.startswith(".")]

    for filename in tqdm(filenames):
        sample, orig_freq = torchaudio.load(src_dir + filename)
        if orig_freq != target_freq:
            print(f"Resampling {filename} from {orig_freq} Hz to {target_freq} Hz...")
            resampled = resample_audio(sample, orig_freq, target_freq)

            if overwrite:
                # Save the resampled audio
                torchaudio.save(src_dir + filename, resampled, target_freq)
                print(f"Completed.")

            else:
                if dest_dir is None:
                    raise ValueError("Destination directory must be specified.")
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                torchaudio.save(dest_dir + filename, resampled, target_freq)
                print(f"Completed.")


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def remove_highly_correlated_features(
    correlation_matrix_comp: pd.DataFrame,
    correlation_matrix_no_target: pd.DataFrame,
    target_variable: str = "label",
    threshold: float = 0.7,
    max_corr_count: int = 4,
) -> List[str]:
    """
    Remove highly correlated features based on correlation matrix.

    Parameters:
        correlation_matrix_comp (pd.DataFrame): Complete correlation matrix including the target variable.
        correlation_matrix_no_target (pd.DataFrame): Correlation matrix without the target variable.
        target_variable (str): The name of the target variable.
        threshold (float): Threshold above which features are considered highly correlated.
        max_corr_count (int): Maximum number of highly correlated features to keep.

    Returns:
        List[str]: List of features to remove.
    """
    features_to_remove = set()

    # Iterate through each feature
    for feature in correlation_matrix_no_target.columns:
        # Count the number of features with correlation higher than threshold
        correlated_count = (correlation_matrix_no_target[feature] > threshold).sum()
        # If the count is greater than the maximum allowed count
        if correlated_count > max_corr_count:
            # Get the correlations of the highly correlated features with the target variable
            target_correlations = correlation_matrix_no_target.loc[feature]
            candidates = target_correlations[target_correlations > threshold].index
            best = (
                correlation_matrix_comp.loc[candidates, target_variable].abs().idxmax()
            )
            features_to_remove.update(candidates.difference([best]))

    # Remove the highly correlated features
    print(f" {len(features_to_remove)} features should be removed")
    return list(features_to_remove)


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def split_col_groups(
    columns: List[str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Split a list of column names into groups based on the number of columns.

    Args:
        columns (List[str]): List of column names.

    Returns:
        Tuple[List[str], List[str], List[str], List[str]]: A tuple containing lists of column names
            divided into groups.
    """
    ncol = len(columns)
    column1: List[str] = []
    column2: List[str] = []
    column3: List[str] = []
    column4: List[str] = []

    if int(ncol // 3) <= 15:
        column1 = columns[: int(ncol // 3)]
        column2 = columns[int(ncol // 3) : int(2 * ncol // 3)]
        column3 = columns[int(2 * ncol // 3) : -1]
        column4 = None
        column1.extend([columns[-1]])
        column2.extend([columns[-1]])
        column3.extend([columns[-1]])
    else:
        column1 = columns[: int(ncol // 4)]
        column2 = columns[int(ncol // 4) : int(2 * ncol // 4)]
        column3 = columns[int(2 * ncol // 4) : int(3 * ncol // 4)]
        column4 = columns[int(3 * ncol // 4) : -1]
        column1.extend([columns[-1]])
        column2.extend([columns[-1]])
        column3.extend([columns[-1]])
        column4.extend([columns[-1]])

    return column1, column2, column3, column4


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def plot_mfcc_to_spectrogram(audio: np.ndarray, ax: plt.Axes, sr: int = 4000, important_mfccs: list = None, title: str = None) -> None:
    """
    Plot the Mel spectrogram of an audio signal along with optional highlighting of important MFCCs.

    Parameters:
    audio (ndarray): The audio signal.
    ax (plt.Axes): The matplotlib axis to plot on.
    sr (int, optional): The sampling rate of the audio signal. Defaults to 4000.
    important_mfccs (list, optional): A list of indices representing the important MFCCs to be highlighted. Defaults to None.
    title (str, optional): The title of the plot. Defaults to None.
    
    Returns:
    None
    """
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    
    # Compute Mel spectrogram
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Plot Mel spectrogram
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set(title=f'MFCC to Spectrogram - {title}')
    ax.figure.colorbar(img, ax=ax, format='%+2.0f dB')
    
    # Highlight regions corresponding to important MFCCs
    if important_mfccs is not None:
        for mfcc in important_mfccs:
            # Map MFCC to corresponding frequency bin
            freq_bin = int(librosa.mel_frequencies(n_mels=128)[mfcc])
            ax.axhline(y=freq_bin, color='w', linestyle='--')


    
    
# --------------------------------------------------------------------
# --------------------------------------------------------------------


def normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalize the given data by scaling it between 0 and 1.

    Parameters:
    data (numpy.ndarray): The input data to be normalized.

    Returns:
    numpy.ndarray: The normalized data.

    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# --------------------------------------------------------------------
# --------------------------------------------------------------------

def plot_waveform_with_mfcc_attributions(audio: np.ndarray, ax: plt.Axes, sr: int = 4000, important_mfccs: list = None, title: str = None) -> None:
    """
    Plot the waveform of an audio signal along with the attributions computed for important MFCCs.

    Parameters:
    audio (ndarray): The audio signal.
    ax (plt.Axes): The matplotlib axis to plot on.
    sr (int): The sample rate of the audio signal (default is 4000).
    important_mfccs (list): A list of indices representing the important MFCCs (default is None).
    title (str): The title of the plot (default is None).

    Returns:
    None
    """
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    
    # Placeholder for attributions (to be computed for the important MFCCs)
    attributions = np.zeros_like(audio)
    
    # Compute attributions (for simplicity, this example uses MFCC magnitudes as attributions)
    for mfcc in important_mfccs:
        # Reshape and stretch MFCC values to match the length of the audio
        mfcc_values = mfccs[mfcc]
        mfcc_stretched = np.interp(np.arange(len(audio)), np.linspace(0, len(audio), len(mfcc_values)), mfcc_values)
        attributions += mfcc_stretched
    
    # Normalize both waveform and attributions
    audio_norm = normalize(audio)
    attributions_norm = normalize(attributions)
    
    # Plot waveform with attributions
    ax.plot(audio_norm, label='Waveform')
    ax.plot(attributions_norm, label='MFCC Attributions', color='r', alpha=0.6)
    ax.legend()
    if title:
        ax.set_title(f'Waveform with MFCC Attributions - {title}')