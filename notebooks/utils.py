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

# ----------------- Functions ---------------------------


def move_corrupted_files(src_dir, dest_dir):
    """
    Check for corrupted files in a directory. If a file is corrupted, it is moved to a new directory.
    It prints the names of corrupted files and the total count of corrupted files found in the directory.

    Args:
        dir (str): The directory path to check for corrupted files.

    Returns:
        count (int): The total count of corrupted files found in the directory.
        filenames (list): The names of corrupted files.

    Prints the names of corrupted files and the total count of corrupted files found in the directory.
    """
    count = 0
    filenames = []
    

    os.makedirs(dest_dir, exist_ok=True)
    
    for file in tqdm(os.listdir(src_dir), desc='Checking files'):
        try:
            _ = torchaudio.load(src_dir + file)
        except:
            # cut the file and paste in a new directory
            os.system(f'mv {src_dir + file} {dest_dir + file}')
            
            print(ValueError(f'File {file} is corrupted'))
            count += 1
            filenames.append(file)
            
    print(f'Files corrupted in {src_dir}:\t{count}\n')
    return count, filenames
    
    
# --------------------------------------------------------------------
# --------------------------------------------------------------------

def slicing(waveform, offset=0, num_frames=None):
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



def extract_features(dir_path, label, frame_length, sample_rate=44100, n_mfcc=13, melkwargs={}):
    """
    Extract features from audio files. It takes the directory path containing the audio files and returns a list of tensors containing features for each audio file.
    Each element in the list is a tensor of shape (num_frames, n_mfcc+6+2), where num_frames is the number of frames in the audio file, n_mfcc is the number of MFCC coefficients,
    6 is the number of additional features (Chroma STFT, RMS, Spectral Centroid, Spectral Bandwidth, Spectral Rolloff, Zero Crossing Rate), and the last two columns are the label and the filename.
    The features are computed on a 1 sec audio segment at a time. As a consequence an audio file with a length of 10 sec will have 10 tensors in the list. Note that if
    the audio file is stereo, it will be converted to mono before computing the features.
    
    Args:
    - dir_path (str): Directory path containing the audio files.
    - sample_rate (int): Desired sample rate for audio processing (default: 44100 Hz).
    - label (int): The label for the audio files (0 for real, 1 for fake).
    - n_mfcc (int): Number of MFCC coefficients to compute (default: 13).
    - melkwargs (dict): Additional arguments for Mel-scale transformation (default: {}).
    
    Returns:
    - features (list): List of tensors containing features for each audio file.
    """
     
    filenames = os.listdir(dir_path)
    filenames = [file for file in filenames if not file.startswith('.')]
    frame_length = int(1/frame_length)
    
#     print(f"Checking the sample rate of the audio files in {dir_path}...")
#     for file in filenames:
#         metadata = torchaudio.info(dir_path + file)
#         # Check the sample rate of the audio. If different from the desired sample rate, raise an error
#         if metadata.sample_rate != sample_rate:
#             raise ValueError(f"Sample rate of the audio {file} is {metadata.sample_rate} Hz. It should be {sample_rate} Hz.\nPlease resample using the provided function 'check_resample_sample_rate'.")

    features = [] # List to store the features for all files
    
    for file in tqdm(filenames, desc=f"Extraction in progress"):
        audio, orig_sample_rate = torchaudio.load(dir_path + file)
        orig_sample_rate_tmp = int(orig_sample_rate/frame_length)
        audio_length = int(max(audio[0].shape) / (orig_sample_rate_tmp)) # Length of the audio in half seconds. Discard the remainder.

        sample_rate = orig_sample_rate
        # Check the sample rate of the audio. If different from the desired sample rate, raise an error
        #if orig_sample_rate != sample_rate:
        #    raise ValueError(f"Sample rate of the audio {file} is {orig_sample_rate} Hz. It should be {sample_rate} Hz.\nPlease resample using the provided function 'check_resample_sample_rate'.")
        
        
        # Reduce the audio from stereo to mono if needed
        if audio.shape[0] > 1:
            print(f"Converting stereo audio to mono for {file}...")
            audio = torch.mean(audio, dim=0).reshape(1, -1)
            #print(f"Shape of the mono audio: {audio_mono.shape}")
        
        features_local = [] # List to store the MFCC features for each file
        
        for i in range(audio_length):
            # Lazy load the audio, one sec at a time, to avoid memory issues
            audio_mono = slicing(audio, offset=int(sample_rate/frame_length*i), num_frames=int(sample_rate/frame_length*(i+1)))
                        
            # get the MFCC features
            mfcc_features_tmp = extract_mfcc(audio_mono, sample_rate, n_mfcc, melkwargs)
            #print(f'MFCC features shape: {mfcc_features_tmp.shape}')
            chroma_stft = extract_chroma_stft(audio_mono, sample_rate)
            #print(f'Chroma STFT shape: {chroma_stft.shape}')
            rms = extract_rms(audio_mono)
            #print(f'RMS shape: {rms.shape}')
            spec_cent = extract_spectral_centroid(audio_mono, sample_rate)
            #print(f'Spectral Centroid shape: {spec_cent.shape}')
            spec_bw = extract_spectral_bandwidth(audio_mono, sample_rate)
            #print(f'Spectral Bandwidth shape: {spec_bw.shape}')
            rolloff = extract_spectral_rolloff(audio_mono, sample_rate)
            #print(f'Spectral Rolloff shape: {rolloff.shape}')
            zcr = extract_zero_crossing_rate(audio_mono)
            #print(f'Zero Crossing Rate shape: {zcr.shape}')
            
            # Concatenate the features
            features_tmp = torch.cat((mfcc_features_tmp, chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr), dim=0)
            features_local.append(features_tmp)
        
        # if features_local is empty, skip the file
        if features_local == []:
            print(f"No features extracted for {file}. Skipping...")
            continue
       
        # Stack the MFCC features into a single tensor
        features_local = torch.stack(features_local, dim=0) 
        # attach the label in the last column
        
        features_local = torch.cat((features_local, torch.ones(features_local.shape[0], 1)*label), dim=1)
        # attach the filename in the last column
        features_local = torch.cat((features_local, torch.full((features_local.shape[0], 1), filenames.index(file))), dim=1)
        
        features.append(features_local)
    
    print('Finished processing all files.\n')
    return features


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def save_features(data, path, name):
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
        print('Features already exist! Please check the directory to avoid overwriting the data.')
        return None
    
    else: 
        print('Saving features...')
        # save the data
        np.savez_compressed(path + name, X=data[:,:-2], y=data[:,-2], filename=data[:,-1])
    
    return None
    

# --------------------------------------------------------------------
# --------------------------------------------------------------------


def extract_mfcc(audio, sample_rate, n_mfcc, melkwargs):
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
    mfcc_transform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=n_mfcc,
    melkwargs=melkwargs
    )
    mfcc_features = mfcc_transform(audio)
    return torch.mean(mfcc_features, dim=2).reshape(-1) # Take the mean of the MFCC coefficients over time. The number of columns depends on the hop_length.
    #return mfcc_features


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def extract_chroma_stft(audio, sample_rate):
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


def extract_rms(audio):
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



def extract_spectral_centroid(audio, sample_rate):
    """
    Extract Spectral Centroid features from an audio file.

    Args:
    - audio (torch.Tensor): The audio file.
    - sample_rate (int): The sample rate of the audio file.

    Returns:
    - spec_cent (torch.Tensor): The Spectral Centroid features of the audio file.
    """
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=audio.numpy(), sr=sample_rate))
    spec_cent = torch.tensor(spec_cent)
    return spec_cent.unsqueeze(0)

# --------------------------------------------------------------------
# --------------------------------------------------------------------


def extract_spectral_bandwidth(audio, sample_rate):
    """
    Extract Spectral Bandwidth features from an audio file.

    Args:
    - audio (torch.Tensor): The audio file.
    - sample_rate (int): The sample rate of the audio file.

    Returns:
    - spec_bw (torch.Tensor): The Spectral Bandwidth features of the audio file.
    """
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=audio.numpy(), sr=sample_rate))
    spec_bw = torch.tensor(spec_bw)
    return spec_bw.unsqueeze(0)


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def extract_spectral_rolloff(audio, sample_rate):
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


def extract_zero_crossing_rate(audio):
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


def generate_samples(dir, target_num, noise_factor, speed_factor, pitch_factor, marker=''):
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
		print(f'No need to generate samples for {dir} directory')
		return
	else:
		percent = to_generate / num_files * 100
		# warn the user about the percentage of the data to be generated
		print(f'Generating {to_generate} samples ({percent:.2f}%) for {dir} directory')

	to_generate_noise = to_generate // 3
	to_generate_speed = to_generate // 3
	to_generate_pitch = to_generate - to_generate_noise - to_generate_speed

	try:
		files_noisy = random.sample(files, k=to_generate_noise) # sample without replacement
	except:
		files_noisy = random.choices(files, k=to_generate_noise) # sample with replacement
		# multiple names can be the same. This may lead to overwriting the files thus not reaching the target number of files.
		
	try:
		files_speed = random.sample(files, k=to_generate_speed)
	except:
		files_speed = random.choices(files, k=to_generate_speed)
	try:
		files_pitch = random.sample(files, k=to_generate_pitch)
	except:
		files_pitch = random.choices(files, k=to_generate_pitch)

	print('Generating noisy audio samples')
	for i, file in enumerate(files_noisy):
		noise_factor_ = random.uniform(noise_factor[0], noise_factor[1])
		audio, sr = torchaudio.load(dir + file)
		audio = audio.mean(0).reshape(1, -1).numpy()[0]
		noise = np.random.randn(len(audio)) * noise_factor_
		noisy_audio = audio + noise
		torchaudio.save(dir + f'noisy_{i}' + marker + file, torch.tensor(noisy_audio).unsqueeze(0), sr)
		# the {i} solves the issue of overwriting the files thus maintaining the desired number of files

	print('Generating speed audio samples')
	for i, file in enumerate(files_speed):
		speed_factor_ = random.uniform(speed_factor[0], speed_factor[1])
		audio, sr = torchaudio.load(dir + file)
		audio = audio.mean(0).reshape(1, -1).numpy()[0]
		audio_speed = librosa.effects.time_stretch(audio, rate=speed_factor_)
		torchaudio.save(dir + f'speed_{i}' + marker + file, torch.tensor(audio_speed).unsqueeze(0), sr)

	print('Generating pitch audio samples')
	for i, file in enumerate(files_pitch):
		pitch_factor_ = random.uniform(pitch_factor[0], pitch_factor[1])
		audio, sr = torchaudio.load(dir + file)
		audio = audio.mean(0).reshape(1, -1).numpy()[0]
		audio_pitch = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_factor_)
		torchaudio.save(dir + f'pitch_{i}' + marker + file, torch.tensor(audio_pitch).unsqueeze(0), sr)

	print('Done!')
 
 
# --------------------------------------------------------------------
# --------------------------------------------------------------------
 
 
def remove_generated_samples(dir, marker):
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

	print(f'Removed {len(generated_files)} generated samples from {dir} directory')
 
 
# --------------------------------------------------------------------
# --------------------------------------------------------------------


def get_move_outliers(src_dir, out_dir, classname, outliers, audio_info, move=False):
    """
    Get outliers for a given class and move them to the outliers folder.

    Parameters:
    src_dir (str): The source directory where the audio files are located.
    classname (str): The name of the class for which outliers are to be retrieved.
    move (bool, optional): Flag indicating whether to move the outliers to the outliers folder. 
                           Defaults to False.

    Returns:
    out (list): A list of filenames of the outliers for the given class.
    """
    
    out = outliers[outliers['label']==classname]['filename'].to_list()
    print(f'{classname} outliers {out}')
    print(f'number of {classname} outliers {len(out)}')

    # drop the outliers for {classname}
    audio_info_outliers = audio_info[~audio_info['filename'].isin(out)]

    print('directory size', len(os.listdir(src_dir)))

    if move == False:
        return out
    elif move == True:
        
        # move the outliers to the outliers folder using shutil
        os.makedirs(out_dir + classname, exist_ok=True)
        for file in out:
            shutil.move(src_dir + file, out_dir + classname)

        print('post-removal directory size', len(os.listdir(src_dir)))
        return out
    else:
        print('Invalid move value. Please enter True or False')
        return None
    
    
# --------------------------------------------------------------------
# --------------------------------------------------------------------