import simfile
import librosa
import numpy as np
import pandas as pd
from simfile.timing import TimingData
from simfile.notes import NoteData
from simfile.notes.timed import time_notes
from keras.utils import np_utils
import random


def _find_chart(charts, stepstype, difficulty):
    for chart in charts:
        if chart.stepstype == stepstype and chart.difficulty == difficulty:
            return chart
    return False


def filter_charts(input_filename, stepstype, difficulty):
    """
    Filters a simfile and returns a single chart for a selected stepstype and difficulty.

    Parameters:

    input_filename (string): Filename of target simfile.
    stepstype (string): Target stepstype for desired chart.
    difficulty (string): Target difficulty for desired chart.

    Returns:

    The desired chart based on stepstype, difficulty and filename.
    """

    smfile = simfile.open(input_filename)
    target = _find_chart(smfile.charts, stepstype, difficulty)
    if target:
        smfile.charts = [target]
        return smfile
    else:
        print(
            f"No simfile of type {stepstype} and difficulty {difficulty} found.")
        return False


def mutate_simfile(input_filename, difficulty, stepstype):
    """

    Changes a simfile by removing all but a single difficulty and chart type.

    Parameters:

    input_filename (string): The file to change.
    difficulty (string): The difficulty to isolate and leave in place.
    dance_type (string): The chart type to isolate and leave in place.

    """
    with simfile.mutate(
        input_filename,
        backup_filename=f'{input_filename}.old',
    ) as sm_file:
        if sm_file.subtitle.endswith('(edited)'):
            raise simfile.CancelMutation
        filter_charts(sm_file, stepstype, difficulty)
        sm_file.subtitle += '(edited)'


def extract_simfile_data(sound_filename, sim_filename, stepstype, difficulty):
    """
    Extracts simfile data and provides a Pandas dataframe with time-sliced chroma data and note data.

    Parameters:

    sound_filename (string): Path to a .ogg file for the song.
    sim_filename (string): Path to the .sm file for the chart.
    stepstype (string): The chart type to target for note data.
    difficulty (string): The difficulty to target for note data.

    Returns:

    A dataframe with sound data and target note type.

    """

    sim = filter_charts(sim_filename, stepstype, difficulty)
    if sim:
        note_data = NoteData(sim.charts[0])
        timing_data = TimingData(sim, sim.charts[0])
        note_time_pairs = []
        for timed_note in time_notes(note_data, timing_data):
            note_time_pairs.append((timed_note.note.column, timed_note.time))
        note_time_pairs = np.array(note_time_pairs)

        sound_file, sr = librosa.load(sound_filename)
        chroma = librosa.feature.chroma_stft(y=sound_file, sr=sr)
        target_map = [4] * len(chroma[0])
        secs_per_sample = 1.0 / sr

        for i in range(len(chroma[0])):
            sample_num = 512 * i
            time_in_song = sample_num * secs_per_sample
            for item in note_time_pairs:
                if (item[1] < (time_in_song + (secs_per_sample * 256))) and (item[1] > (time_in_song - (secs_per_sample * 256))):
                    target_map[i] = item[0]

        chroma = chroma.T
        target_map = np.array(target_map)
        output_df = pd.DataFrame(chroma)
        output_df["target"] = target_map

        return output_df
    return False


def cnn_xy_split(input_df):
    """
    Takes a dataframe and splits it into a format ingestible by a Keras CNN.

    Parameters:

    input_df (pandas.DataFrame): 13 by n dataframe of audio and note data, where n is dependent
    on the length of audio and chart.

    Returns:

    X matrix with audio data and Y matrix with one-hot-encoded target note data.

    """
    x_out = input_df.loc[:, input_df.columns.difference(
        ["target"])].to_numpy()
    x_out = np.reshape(x_out, (x_out.shape[0], x_out.shape[1], 1))
    y_out = input_df["target"].to_numpy()
    y_out = np_utils.to_categorical(y_out)

    return x_out, y_out


def knn_xy_split(input_df):
    """
    Takes a dataframe and splits it into a format ingestible by an sklearn KNN model.

    Parameters:

    input_df (pandas.DataFrame): 13 by n dataframe of audio and note data, where n is dependent
    on the length of audio and chart.

    Returns:

    X matrix with audio data and Y matrix with target note data.

    """

    x_out = input_df.loc[:, input_df.columns.difference(
        ["target"])].to_numpy()
    y_out = input_df["target"].to_numpy()

    return x_out, y_out


def filter_training_data(input_df):

    yes_note = input_df[input_df.target != 4].copy()

    fourth = int(len(yes_note.index))

    drop_list = input_df.index[input_df["target"] == 4].tolist()
    input_df = input_df.drop(drop_list[fourth:], axis=0)

    return input_df
