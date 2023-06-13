import torch
import numpy as np
import pretty_midi
import os
import pandas as pd
import requests
import zipfile
import pickle

from src.preprocessing import data_preprocessing


def accuracy(y_true, y_pred):
    """
    Model의 accuracy
    """
    y_true = torch.argmax(y_true, axis=2)  # y_true: (batch_size, 64)
    y_pred = torch.argmax(y_pred, axis=2)
    total_num = y_true.shape[0] * y_true.shape[1]

    return torch.sum(y_true == y_pred) / total_num


def prob_label(prob):
    """
    model에서 생성된 확률분포 prob을 반영해서
    prob[idx] 번 째에서의 소리(9가지 중 하나)를 라벨링
    """
    num_classes = prob.shape[1]
    play = np.zeros(prob.shape)
    for seq in range(prob.shape[0]):
        label = np.random.choice(num_classes, size=1, p=prob[seq])
        play[seq, label] = 1

    return play


def generate_midi_file(roll, fs, comp=9):
    """
    prob_label을 통해 표시된 드럼 소리를 미디파일로 변환
    """
    fs_time = 1 / fs
    standard = {
        35: "kick",
        38: "snare",
        46: "open hi-hat",
        42: "closed hi-hat",
        50: "high tom",
        48: "mid tom",
        45: "low tom",
        49: "crash",
        51: "ride",
    }

    encoded = {
        "kick": 0,
        "snare": 1,
        "open hi-hat": 2,
        "closed hi-hat": 3,
        "high tom": 4,
        "mid tom": 5,
        "low tom": 6,
        "crash": 7,
        "ride": 8,
    }
    reverse_standard = {v: k for k, v in standard.items()}
    reverse_encoded = {v: k for k, v in encoded.items()}

    decimal_idx = np.where(roll == 1)[1]
    binary_idx = list(map(lambda x: np.binary_repr(x, comp), decimal_idx))

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=119, is_drum=True)
    pm.instruments.append(inst)

    for i, drum_sound in enumerate(binary_idx):
        start_time = fs_time * i
        end_time = fs_time * (i + 1)

        for j in range(0, len(drum_sound)):
            if drum_sound[j] == "1":
                pitch = reverse_standard[reverse_encoded[j]]
                inst.notes.append(pretty_midi.Note(80, pitch, start_time, end_time))

    return pm


def prepare_data():
    if os.path.isfile("./data/midi_data.pkl"):
        with open("./data/midi_data.pkl", "rb") as f:
            data = pickle.load(f)
    elif os.path.isfile("./data/groove"):
        info = pd.read_csv("./data/groove/info.csv")
        file_list = info.midi_filename
        data = data_preprocessing(file_list)
    else:
        # groove 데이터가 없는 경우 다운로드
        url = "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip"
        response = requests.get(url)
        filename = "groove-v1.0.0-midionly.zip"
        with open(filename, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall("./data")
        # 다운로드한 ZIP 파일 삭제
        os.remove(filename)
        info = pd.read_csv("./data/groove/info.csv")
        file_list = info.midi_filename
        data = data_preprocessing(file_list)

        return data
