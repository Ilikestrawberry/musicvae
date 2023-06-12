import os
import torch
import pickle
import pandas as pd
from datetime import datetime
import argparse
from omegaconf import OmegaConf

from torch.utils.data import DataLoader

from src.preprocessing import data_preprocessing
from src.dataloader import GrooveDataset
from src.model import MusicVAE
from src.utils import generation_midi_file, prob_label
from src.trainer import Trainer

global device

if torch.backends.mps.is_available():
    device = torch.device("mps:0")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


if os.path.isfile("./data/midi_data.pkl"):
    with open("./data/midi_data.pkl", "rb") as f:
        data = pickle.load(f)
else:
    info = pd.read_csv("./data/groove/info.csv")
    file_list = info.midi_filename
    data = data_preprocessing(file_list)


n_data = len(data)
train_set = GrooveDataset(data[: int(n_data * 0.9)])
dev_set = GrooveDataset(data[int(n_data * 0.9) :])

print(f"{n_data} data loaded")
print("train_set: ", len(train_set))
print("dev_set: ", len(dev_set))


def train(conf):
    train_dataloader = DataLoader(train_set, batch_size=conf.train.batch_size)
    dev_dataloader = DataLoader(dev_set, batch_size=conf.train.batch_size)

    datalength = len(train_set)  # Trainer에서 Onecycle 스케줄러 인자 전달을 위한 값
    model = MusicVAE(conf).to(device)

    trainer = Trainer(conf, model, datalength)
    trainer.train(train_dataloader, dev_dataloader)


def generate(conf):
    now = datetime.now()
    generated_time = now.strftime("%d-%H-%M")
    saved_model = os.path.join(conf.path.load_model_dir, conf.path.load_model_file)

    model = MusicVAE(conf)
    model.load_state_dict(torch.load(saved_model))
    model.to(device)

    pred = model.generate()

    pred = pred.squeeze(0).numpy()
    generated_midi = prob_label(pred)
    pm = generation_midi_file(generated_midi, fs=8)
    file_path = os.path.join(conf.path.load_model_dir, f"{generated_time}_generated.midi")
    pm.write(file_path)
    print("MIDI file is generated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 여기서 omegaconfig 파일 이름 설정하고 실행해주세요.(default = base_config.yaml)
    parser.add_argument("--config", "-c", type=str, default="base_config", help="config file name")
    parser.add_argument("--mode", "-m", required=True)

    args = parser.parse_args()
    conf = OmegaConf.load(f"./config/{args.config}.yaml")

    # 터미널 실행 예시 : python main.py -mt -> train 실행
    #                python main.py -mg -> generate 실행

    print("실행 중인 config file: ", args.config)
    if args.mode == "train" or args.mode == "t":
        train(conf)

    elif args.mode == "generate" or args.mode == "g":
        if conf.path.load_model_dir is None:
            print("로드할 모델의 경로를 입력해주세요.")
        else:
            generate(conf)

    else:
        print("실행모드를 다시 입력해주세요.")
        print("train        : t,\ttrain")
        print("generate    : g,\tgenerate")

# batch_size, epoch, lr, save_epoch, wandb on/off, loss_func
# generate_path
