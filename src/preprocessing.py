import numpy as np
from tqdm import tqdm
import pretty_midi
from scipy.sparse import coo_matrix
import os
import pickle

DRUM2MIDI = {
    "kick": [36],
    "snare": [38, 40, 37],
    "high tom": [48, 50],
    "low-mid-tom": [45, 47],
    "high floor tom": [43, 58],
    "open hi-hat": [46, 26],
    "closed hi-hat": [42, 22, 44],
    "crash": [49, 55, 57, 52],
    "ride": [51, 59, 53],
}
IDX2MIDI = {
    0: 36,  # kick
    1: 38,  # snare
    2: 50,  # high tom
    3: 47,  # low-mid tom
    4: 43,  # high floor tom
    5: 46,  # open hi-hat
    6: 42,  # closed hi-hat
    7: 49,  # Crash Cymbal
    8: 51,  # Ride Cymbal
}

DRUM2IDX = {drum: i for i, drum in enumerate(DRUM2MIDI.keys())}
MIDI2IDX = {midi_num: DRUM2IDX[drum] for drum, midi_nums in DRUM2MIDI.items() for midi_num in midi_nums}


def check_time_signature(pm, num=4, denom=4):
    """
    midi data에 담겨있는 박자가 전부 4/4 박자인지 확인
    """
    sign_list = pm.time_signature_changes

    # 데이터가 비어있는 경우 제외
    if len(sign_list) == 0:
        return False

    for sign in sign_list:
        if sign.numerator != num or sign.denominator != denom:
            return False

    return True


def change_fs(beats, target_beats=16):
    """
    4박(1bar)에 16비트를 담기 위한 샘플링 속도 계산
    ex) change_fs = 8 => 1초에 8비트

    논문 figure 2의 output에서 1bar 당 16비트이기 때문에 target_beats = 16으로 설정
    """
    quarter_length = beats[1] - beats[0]
    changed_length = (quarter_length * 4) / target_beats
    change_fs = 1 / changed_length

    return change_fs


def quantize_drum(inst, fs, start_time, comp=9):
    """
    inst(pm)와 sampling rate를 기준으로
    드럼 소리/시간 정보를 담고있는 drum_roll을 반환
    """
    fs_time = 1 / fs
    end_time = inst.get_end_time()

    # 비트가 들어갈 수 있는 time stamp
    quantize_time = np.arange(start_time, end_time + fs_time, fs_time)
    drum_roll = np.zeros((quantize_time.shape[0], comp))  # shape: (비트 들어갈 수 있는 time stamp, 드럼 소리 가지 수(9)

    for _, note in enumerate(inst.notes):
        # pitch 번호가 dic에 없는 경우엔 패스
        if note.pitch not in MIDI2IDX.keys():
            continue
        # 각 note가 시작/종료되는 시점에서 가장 가까운 quntize_time을 인덱싱
        start_idx = np.argmin(np.abs(quantize_time - note.start))
        end_idx = np.argmin(np.abs(quantize_time - note.end))

        # 시작/종료 인덱스가 같은 경우엔 임의로 end_idx += 1
        if start_idx == end_idx:
            end_idx += 1

        range_idx = np.arange(start_idx, end_idx, 1)
        inst_idx = MIDI2IDX[note.pitch]

        for idx in range_idx:
            # idx 번 째 time stamp에 inst_idx 소리를 입력
            drum_roll[idx, inst_idx] = 1
    return drum_roll


def windowing(roll, window_size=64, bar=16, cut_ratio=0.9):
    """
    데이터를 일정한 길이(window_size)로 자르고
    비트 정보가 90% 넘게 존재하는 경우에만 사용

    output shape: (num_windows, 64, 9)
    """
    new_roll = []
    # roll.shape[0]은 비트가 들어갈 수 있는 time stamp 개수
    num_windows = roll.shape[0] // window_size
    do_nothing = np.sum((roll == 0), axis=1) == roll.shape[1]

    for i in range(0, num_windows):
        break_flag = False
        start_idx = window_size * i
        end_idx = window_size * (i + 1)

        check_empty = do_nothing[start_idx:end_idx]
        for j in range(0, window_size, bar):
            if np.sum(check_empty[j : j + bar]) > (bar * cut_ratio):
                break_flag = True
                break
        if break_flag:
            continue

        new_roll.append(np.expand_dims(roll[start_idx:end_idx], axis=0))

    return np.vstack(new_roll)


def binary_to_decimal(roll):
    """
    이진수로 저장되어 있는 midi 데이터를 십진수로 변환
    """
    decimal = 0
    length = roll.shape[0]

    for i, bi in enumerate(roll):
        decimal += np.power(2, length - i - 1) * bi

    # one_hot_encoding()에서 인덱스로 사용되기 때문에 int로 반환
    return int(decimal)


def one_hot_encoding(roll):
    """
    quantize_drum()에서 라벨링된 드럼 소리를 원핫인코딩 형태로 변환

    input shape: (num_windows, 64 ,9)
    output shape: (num_windows, 64, 512(2**9))
    """
    last_axis = len(roll.shape) - 1
    I = np.eye(np.power(2, roll.shape[-1]), dtype="bool")
    dec_idx = np.apply_along_axis(binary_to_decimal, last_axis, roll)

    return I[dec_idx]


def transform_to_midi(roll, fs, comp=9):
    """
    생성된 roll을 midi 정보로 변환
    """
    fs_time = 1 / fs
    decimal_idx = np.where(roll == 1)[1]
    print(roll.shape)
    binary_idx = list(map(lambda x: np.binary_repr(x, comp), decimal_idx))

    # midi 객체
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=119, is_drum=True)  # 119: Synth Drum
    pm.instruments.append(inst)

    for i, drum_sound in enumerate(binary_idx):
        start_time = fs_time * i
        end_time = fs_time * (i + 1)

        for j in range(0, len(drum_sound)):
            if drum_sound[j] == "1":
                pitch = IDX2MIDI[j]
                inst.notes.append(pretty_midi.Note(80, pitch, start_time, end_time))  # 80: velocity(세기)

    return pm


def data_preprocessing(file_list):
    """
    groove 파일을 window_size 길이로 나누고 pickle 파일로 변환
    """
    data = []
    print("Data preprocessing...")

    for f_name in tqdm(file_list):
        file_path = "./data/groove/" + f_name
        try:
            # 파일을 midi 형태로 로드
            pm = pretty_midi.PrettyMIDI(file_path)

            # 4/4박자 확인
            ts = pm.time_signature_changes
            if len(ts) == 0:
                time_signature = (4, 4)
            else:
                ts = ts[0]
                time_signature = (ts.numerator, ts.denominator)
            if time_signature != (4, 4):
                continue

            # get_onsets: 첫음이 시작되는 시간 / get_beats: 비트가 찍힌 시간
            start_time = pm.get_onsets()[0]
            beats = pm.get_beats(start_time)
            fs = change_fs(beats)

            for inst in pm.instruments:
                if inst.is_drum == True:
                    drum_roll = quantize_drum(inst, fs, start_time)
                    drum_roll = windowing(drum_roll)
                    drum_roll = one_hot_encoding(drum_roll)
                    for i in range(0, drum_roll.shape[0]):
                        data.append(coo_matrix(drum_roll[i]))  # 희소배열을 coordinate(저용량)
        except:
            continue

    os.makedirs("./data", exist_ok=True)
    with open("./data/midi_data.pkl", "wb") as f:
        print(f"{len(data)} files saved!!")
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return data
