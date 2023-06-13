# MusicVAE

## 논문 요약
- 기존의 RNN 구조는 직전 데이터의 영향을 많이 받고 오래 전 데이터의 영향은 갈수록 줄어드는 문제점이 있음.
- 계층적인(Hierarchical) 구조를 도입해 가까운 정보 뿐만 아니라 데이터 전체적인 정보를 반영할 수 있도록 함.
- 이를 위해서 Variational Autoencoder(VAE) 구조를 활용하고 Encoder, Decoder 이외에 Conductor 구조를 추가.
### 과정
- Encoder -> Conductor 과정: 실제분포 p(z)와 잠재분포 q(z)가 최대한 비슷해지도록 파라미터(λ) 학습.
- Conductor -> Decoder 과정: q(z)에서 생성된 값을 decoder에 입력하여 encoder 입력값인 x와 decoder에서 생성된 x'의 차이가 최소화 되도록 파라미터(θ) 학습.

## 데이터 전처리
- "groove-v1.0.0-midionly.zip" 데이터를 사용.
1. 파일에 담겨있는 정보가 4/4박자인지 확인.
2. 파일의 Note 정보(드럼 소리)를 적절한 위치(시간)에 인덱싱.
3. 통일된 입력을 위해 일정한 길이(64)로 데이터를 나눔.
4. 9가지 드럼 소리를 원-핫 인코딩으로 변환(512).
5. 하나의 pickle 파일로 저장.

## 학습
- Optimizer: Adam
- Scheduler: 

## 생성
