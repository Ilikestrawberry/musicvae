import torch
from torch import nn
import torch.nn.functional as F


if torch.backends.mps.is_available():
    device = torch.device("mps:0")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class MusicVAE(nn.Module):
    def __init__(self, conf):
        super().__init__()
        """
        Encoder(Bi-LSTM)
        Conductor(LSTM)
        Decoder(LSTM)
        
        model의 파라미터 수는 논문에 나온 숫자를 그대로 사용
        dropout은 임의로 추가
        """

        self.input_size = 512
        self.bar_units = 16  # 미니멈 16분 음표 기준

        # encoder params
        self.enc_hidden_size = 2048
        self.enc_latent_dim = 512
        # conductor params
        self.con_dim = 512
        self.con_hidden_size = 1024
        # decoder params
        self.dec_hidden_size = 1024

        # 임의로 dropout 추가
        self.dropout = conf.train.dropout

        self.encoder = Encoder(self.input_size, self.enc_hidden_size, self.enc_latent_dim, dropout=self.dropout)
        self.conductor = Conductor(self.input_size, self.con_dim, self.con_hidden_size, dropout=self.dropout)
        self.decoder = Decoder(self.input_size, self.dec_hidden_size, dropout=self.dropout)
        self.num_hidden = self.decoder.num_hidden

    def forward(self, x):
        """
        1. input(x)을 받은 encoder에서 잠재분포(latent_z)와 평균(mu), 편차(std)를 반환
        2. z를 전달받은 conductor에서 decoder의 입력을 샘플링
        3. 입력값을 받은 decoder에서 x'을 출력

        input shape: (batch_size, seq_len(64), 512)
        output shape: (batch_size, seq_len, 512)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        latent_z, mu, std = self.encoder(x)
        feat = self.conductor(latent_z)  # feat shape: (batch_size, 4, 512(con_dim))

        x_input = torch.zeros((batch_size, 1, x.shape[2]), device=device)
        x_label = torch.zeros(x.shape[:-1], device=device)
        x_prob = torch.zeros(x.shape, device=device)

        for i in range(seq_len):
            bar_idx = i // self.bar_units  # bar_idx 번 째 박자
            bar_note_idx = i % self.bar_units  # bar_change_idx 번 째 음표
            z = feat[:, bar_idx, :]
            if bar_note_idx == 0:  # 각 박자의 첫 시작은 conductor에서 생성된 z(feat)와 x_input을 기반으로 생성(fig 2)
                h = z.repeat(self.num_hidden, 1, int(self.dec_hidden_size / z.shape[1]))
                c = z.repeat(self.num_hidden, 1, int(self.dec_hidden_size / z.shape[1]))
            label, prob, h, c = self.decoder(x_input, h, c, z)
            # label shape: (batch_size, 1)  // prob shape: (batch_size, 1, 512)

            x_input = x[:, i, :].unsqueeze(1)
            x_label[:, i] = label.squeeze()
            x_prob[:, i, :] = prob.squeeze()

        return x_prob, mu, std

    def generate(self, bar_units=16, seq_len=64):
        """
        표준정규분포 z를 입력으로 conductor와 decoder를 거쳐 midi 생성

        z shape: (1, 512)
        outputs shape: (1, 64, 512)
        """
        z = torch.empty((1, 512)).normal_(mean=0, std=1).to(device)
        feat = self.conductor(z)
        batch_size = 1
        hidden_size = self.decoder.hidden_size
        output_size = self.decoder.input_size

        inputs = torch.zeros((batch_size, 1, output_size), device=device)
        outputs = torch.zeros((batch_size, seq_len, output_size), device=device)

        for i in range(seq_len):
            bar_idx = i // bar_units
            bar_note_idx = i % bar_units
            z = feat[:, bar_idx, :]
            if bar_note_idx == 0:
                h = z.repeat(self.num_hidden, 1, int(hidden_size / z.shape[1]))
                c = z.repeat(self.num_hidden, 1, int(hidden_size / z.shape[1]))

                label, prob, h, c = self.decoder(inputs, h, c, z)
            outputs[:, i, :] = prob.squeeze()

            # decoder의 input shape가 (batch_size, 1, 512)이기 때문에
            # (batch_size, 1) => (batch_size, 1, 512) 형태의 원핫 인코딩으로 다시 변환
            inputs = F.one_hot(label, num_classes=output_size)

        return outputs.detach().cpu()


class Encoder(nn.Module):
    """
    midi 데이터를 학습해 잠재분포(latent_z)를 근사
    """

    def __init__(self, input_size, hidden_size, latent_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_direction = 2  # Bi-LSTM

        self.encoder = nn.LSTM(
            batch_first=True,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=True,
        )
        self.mu = nn.Linear(self.hidden_size * self.num_layers * self.num_direction, self.latent_dim)
        self.std = nn.Linear(self.hidden_size * self.num_layers * self.num_direction, self.latent_dim)
        # input 파라미터 정규화
        self.norm = nn.LayerNorm(self.latent_dim, elementwise_affine=False)

    def forward(self, x):
        x, (h, c) = self.encoder(x)
        h = h.transpose(0, 1).reshape(-1, self.hidden_size * self.num_layers * self.num_direction)

        mu = self.norm(self.mu(h))  # mu shape: (batch_size, latent_dim)
        std = nn.Softplus()(self.std(h))  # 논문 Eq 7
        eps = torch.randn_like(std)
        # z = mu + std를 그대로 사용하는 경우 z를 얻기 위해서 x를 항상 새로 샘플링해야 하는 문제가 발생
        # 표준정규분포 eps를 도입해서 z와 x 사이 샘플링 필수 조건을 없도록 함
        z = mu + (std * eps)  # 논문 Eq 2

        return z, mu, std


class Conductor(nn.Module):
    """
    Encoder에서 얻은 잠재분포(latent_z)를 기반으로 각 박자 첫 부분(feat) 생성
    """

    def __init__(self, input_size, con_dim, hidden_size, num_layers=2, bar=4, dropout=0.2):
        super().__init__()

        self.bar = bar
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.con_dim = con_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_direction = 1  # 단방향

        self.linear = nn.Linear(self.hidden_size, self.con_dim)
        self.conductor = nn.LSTM(
            batch_first=True,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=False,
        )

    def init_hidden(self, z):
        """
        Conductor의 첫 입력값 생성
        """
        h0 = z.repeat(self.num_direction * self.num_layers, 1, int(self.hidden_size / z.shape[1]))
        c0 = z.repeat(self.num_direction * self.num_layers, 1, int(self.hidden_size / z.shape[1]))

        return h0, c0

    def forward(self, z):
        batch_size = z.shape[0]
        h, c = self.init_hidden(z)  # h(c) shape: (num_direction * num_layers, batch_size, latent_dim)
        z = z.unsqueeze(1)

        feat = torch.zeros(batch_size, self.bar, self.hidden_size, device=device)
        z_input = z
        # 각 박자의 첫 시작
        for i in range(self.bar):
            z_input, (h, c) = self.conductor(z_input, (h, c))
            feat[:, i, :] = z_input.squeeze()
            z_input = z
        feat = self.linear(feat)
        # feat shape: (batch_size, 4(bar), con_dim)
        return feat


class Decoder(nn.Module):
    """
    Conductor에서 얻은 초기값과 이전 note를 사용하여 다음 note 생성
    """

    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_direction = 1
        self.num_hidden = self.num_direction * num_layers

        self.logits = nn.Linear(self.hidden_size, self.input_size)
        self.decoder = nn.LSTM(
            batch_first=True,
            input_size=self.input_size + self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=False,
        )

    def forward(self, x, h, c, z):
        # x shape: (batch_size, 1, 512*2) => 직전 note와 conductor에서 생성된 z를 합쳐서 입력으로 사용
        x = torch.cat((x, z.unsqueeze(1)), 2)
        x, (h, c) = self.decoder(x, (h, c))
        logits = self.logits(x)  # logits shape: (batch_size, 1, 512)
        prob = nn.Softmax(dim=2)(logits)  # prob shape: (batch_size 1, 512(output_size))
        output = torch.argmax(prob, 2)  # output shape: (batch_size, 1)

        return output, prob, h, c
