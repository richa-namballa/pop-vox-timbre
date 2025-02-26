import numpy as np


def normalize_data(x):
    """
    Normalize an array between -1 and 1.

    :param x: (np.array) data to normalize
    :return: (np.array) normalized array
    """
    # center data with a mean of 0 and a standard deviation of 1
    z = (x - np.mean(x)) / np.std(x)
    
    # scale between -1 and 1
    q = 2 * ((z - np.min(z)) / (np.max(z) - np.min(z))) - 1
    
    return q


def trim_audio(y, sr, duration):
    """
    Trim an audio array from the beginning to the maximum duration.

    :param y: (np.array) audio array
    :param sr: (int) sample rate of audio array
    :param duration: (float) maximum duration in seconds
    :return: (np.array) trimmed audio array
    """
    # get the duration in samples
    dur_in_samples = int(duration * sr)

    if len(y.shape) == 2:
        # stereo
        # get number of samples
        _ , num_samples = y.shape

        if dur_in_samples < num_samples:
            # if the audio is longer than the specified trim duration
            # trim the audio
            y_trim = y[:, :dur_in_samples]
        else:
            y_trim = y
    elif len(y.shape) == 1:
        # mono
        # get the number of samples
        num_samples = len(y)

        if dur_in_samples < num_samples:
            # if the audio is longer than the specified trim duration
            # trim the audio
            y_trim = y[:dur_in_samples]
        else:
            y_trim = y
    else:
        raise ValueError("Audio array can only be 1-dimensional or 2-dimensional!")

    return y_trim


def fade_in_out(audio, sr, duration=0.5):
    """
    Apply a linear fade in and out to audio data.

    :param audio: (np.array) audio array
    :param sr: (int) sample rate of audio
    :param duration: (float) length of fade in seconds
    :return: (np.array) audio array with fade applied
    """
    if len(audio.shape) == 2:
        num_channels, num_samples = audio.shape
    elif len(audio.shape) == 1:
        num_channels = 1  # mono
        num_samples = len(audio)
    else:
        raise ValueError("Audio array can only be 1-dimensional or 2-dimensional!")

    fade_length = int(duration * sr)  # length of fade in samples

    # create fade curve
    fade_in = np.linspace(0.0, 1.0, fade_length)
    fade_out = np.linspace(1.0, 0.0, fade_length)
    full_amp = np.ones((num_samples - 2 * fade_length))
    fade_curve = np.concatenate((fade_in, full_amp, fade_out))

    # apply fade curve to audio
    y_faded = audio * fade_curve

    return y_faded
 