import os
from datetime import datetime
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch import mps  # torch 버그 때문에 별도로 지정해서 임포트
import wandb

from src.loss import ELBO_loss
from src.utils import accuracy

if torch.backends.mps.is_available():
    device = torch.device("mps:0")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Trainer:
    """
    config를 전달받은 Trainer
    """

    def __init__(self, conf, model):
        self.model = model
        self.init_model_parameters(self.model)

        self.max_epoch = conf.train.max_epoch
        self.batch_size = conf.train.batch_size
        self.lr = conf.train.learning_rate
        self.max_len = 64
        self.criterion = conf.train.loss_func
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = MusicVaeScheduler(optimizer=self.optimizer, learning_rate=self.lr, decay_rate=0.9999, min_learning_rate=0.00001)
        self.eval_func = accuracy

        now = datetime.now()
        self.train_start_time = now.strftime("%d-%H-%M")
        self.save_model_dir = conf.path.save_model_dir
        self.save_period = conf.train.save_period
        self.eval_period = conf.train.eval_period

        self.wandb = conf.train.wandb

        if self.wandb == True:
            wandb.init(project="musicvae")
            wandb.watch(self.model, self.criterion, log="all", log_freq=10)

    def init_model_parameters(self, model):
        """
        초기 모델 파라미터를 초기화
        """
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def save(self, file_name="default"):
        save_dir = os.path.join(self.save_model_dir, self.train_start_time)
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"{file_name}_{self.batch_size}.pt"
        file_path = os.path.join(save_dir, file_name)
        torch.save(self.model.state_dict(), file_path)

    def train_one_epoch(self, dataloader, epoch):
        """
        1 epoch마다 수행하는 작업
        """
        iter_bar = tqdm(dataloader)
        avg_loss = []
        for batch in iter_bar:
            iter_bar.desc
            batch = batch.to(device)
            self.optimizer.zero_grad()
            pred, mu, std = self.model(batch)
            if self.criterion == "elbo":
                loss = ELBO_loss(pred, batch, mu, std, epoch)
            avg_loss.append(loss.item())
            acc = self.eval_func(pred, batch)
            iter_bar.set_description("Train Iter (acc=%5.5f, loss=%5.3f, lr=%5.5f)" % (acc.item(), loss.item() / self.batch_size, self.optimizer.param_groups[0]["lr"]))
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if self.wandb == True:
                wandb.log(
                    {
                        "epoch": epoch,
                        "accuracy": acc,
                        "train loss": loss,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                )

            if device == "mps0":
                mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

        if self.wandb == True:
            wandb.log({"epoch": epoch, "train avg loss": sum(avg_loss) / len(avg_loss)})

    def train(self, train_loader, dev_loader=None):
        """
        model 학습
        train과 model save로 구성
        """
        self.model.train()
        for epoch in range(1, self.max_epoch + 1):
            self.train_one_epoch(train_loader, epoch)
            if dev_loader and (epoch % self.eval_period == 0):
                with torch.no_grad():
                    eval_loss, eval_acc = self.evaluate(dev_loader)
                    if self.wandb == True:
                        wandb.log(
                            {
                                "epoch": epoch,
                                "eval loss": eval_loss,
                                "eval ACC": eval_acc,
                            }
                        )

            if (epoch % self.save_period) == 0:
                file_name = f"epoch_{epoch}"
                self.save(file_name)
                print(f"{epoch} model saved!")

        file_name = f"epoch_{epoch}"
        self.save(file_name)
        print("Final epoch model saved!")

    def evaluate(self, dev_loader):
        """
        dev set을 사용해서 모델의 성능을 평가
        """
        self.model.eval()
        iter_bar = tqdm(dev_loader)

        losses, accuracies = [], []
        for i, batch in enumerate(iter_bar):
            batch = batch.to(device)
            pred, mu, std = self.model(batch)
            pred = pred
            if self.criterion == "elbo":
                loss = ELBO_loss(pred, batch, mu, std, self.max_epoch)
            losses.append(loss.item())

            acc = self.eval_func(pred, batch)
            accuracies.append(acc)
            iter_bar.set_description("Eval Iter (acc=%5.5f, loss=%5.3f)" % (acc.item(), loss.item() / self.batch_size))

            if device == "mps0":
                mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

        return sum(losses) / len(losses), sum(accuracies) / len(accuracies)


class MusicVaeScheduler(_LRScheduler):
    """
    논문에서 사용한 스케줄러
    """

    def __init__(self, optimizer, learning_rate=0.001, decay_rate=0.9999, min_learning_rate=0.00001):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.min_learning_rate = min_learning_rate
        super().__init__(optimizer)

    def get_lr(self):
        return [(self.learning_rate - self.min_learning_rate) * (self.decay_rate**self._step_count) + self.min_learning_rate]
