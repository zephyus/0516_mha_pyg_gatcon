# Networked Multi-agent RL (NMARL)
This repo implements the state-of-the-art MARL algorithms for networked system control, with observability and communication of each agent limited to its neighborhood. For fair comparison, all algorithms are applied to A2C agents, classified into two groups: IA2C contains non-communicative policies which utilize neighborhood information only, whereas MA2C contains communicative policies with certain communication protocols.

Available IA2C algorithms:
* PolicyInferring: [Lowe, Ryan, et al. "Multi-agent actor-critic for mixed cooperative-competitive environments." Advances in Neural Information Processing Systems, 2017.](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)
* FingerPrint: [Foerster, Jakob, et al. "Stabilising experience replay for deep multi-agent reinforcement learning." arXiv preprint arXiv:1702.08887, 2017.](https://arxiv.org/pdf/1702.08887.pdf)
* ConsensusUpdate: [Zhang, Kaiqing, et al. "Fully decentralized multi-agent reinforcement learning with networked agents." arXiv preprint arXiv:1802.08757, 2018.](https://arxiv.org/pdf/1802.08757.pdf)


Available MA2C algorithms:
* DIAL: [Foerster, Jakob, et al. "Learning to communicate with deep multi-agent reinforcement learning." Advances in Neural Information Processing Systems. 2016.](http://papers.nips.cc/paper/6042-learning-to-communicate-with-deep-multi-agent-reinforcement-learning.pdf)
* CommNet: [Sukhbaatar, Sainbayar, et al. "Learning multiagent communication with backpropagation." Advances in Neural Information Processing Systems, 2016.](https://arxiv.org/pdf/1605.07736.pdf)
* NeurComm: Inspired from [Gilmer, Justin, et al. "Neural message passing for quantum chemistry." arXiv preprint arXiv:1704.01212, 2017.](https://arxiv.org/pdf/1704.01212.pdf)

Available NMARL scenarios:
* ATSC Grid: Adaptive traffic signal control in a synthetic traffic grid.
* ATSC Monaco: Adaptive traffic signal control in a real-world traffic network from Monaco city.
* CACC Catch-up: Cooperative adaptive cruise control for catching up the leadinig vehicle.
* CACC Slow-down: Cooperative adaptive cruise control for following the leading vehicle to slow down.

## Requirements
* Python3 == 3.5
* [PyTorch](https://pytorch.org/get-started/locally/) == 1.4.0
* [Tensorflow](http://www.tensorflow.org/install) == 2.1.0 (for tensorboard) 
* [SUMO](http://sumo.dlr.de/wiki/Installing) >= 1.1.0

## Usages
First define all hyperparameters (including algorithm and DNN structure) in a config file under `[config_dir]` ([examples](./config)), and create the base directory of each experiement `[base_dir]`. For ATSC Grid, please call [`build_file.py`](./envs/large_grid_data) to generate SUMO network files before training.

1. To train a new agent, run
~~~
python3 main.py --base-dir [base_dir] train --config-dir [config_dir]
python3 main.py --base-dir real_a1/ma2c_nclm/ --port 100 train --config-dir config/config_ma2c_nclm_net.ini
python3 main.py --base-dir training/real/direction_aw001c06/ma2c_nclm/ --port 189 train --config-dir config/config_ma2c_nclm_net.ini
python3 main.py --base-dir training/grid/direction_aw000c02/ma2c_nclm/ --port 110 train --config-dir config/config_ma2c_nclm_grid.ini
~~~
Training config/data and the trained model will be output to `[base_dir]/data` and `[base_dir]/model`, respectively.

2. To access tensorboard during training, run
~~~
tensorboard --logdir=[base_dir]/log
~~~

3. To evaluate a trained agent, run
~~~
python3 main.py --base-dir [base_dir] evaluate --evaluation-seeds [seeds]
~~~
Evaluation data will be output to `[base_dir]/eva_data`. Make sure evaluation seeds are different from those used in training.    

4. To visualize the agent behavior in ATSC scenarios, run
~~~
python3 main.py --base-dir [base_dir] evaluate --evaluation-seeds [seed] --demo
~~~
It is recommended to use only one evaluation seed for the demo run. This will launch the SUMO GUI, and [`view.xml`](./envs/large_grid_data) can be applied to visualize queue length and intersectin delay in edge color and thickness. 

## Reproducibility
The paper results are based on an out-of-date SUMO version 0.32.0. We are re-running the experiments with SUMO 1.2.0 and will update the results soon. The pytorch impelmention is avaliable at branch [pytorch](https://github.com/cts198859/deeprl_network/tree/pytorch).

## Citation
For more implementation details and underlying reasonings, please check our paper [Multi-agent Reinforcement Learning for Networked System Control](https://openreview.net/forum?id=Syx7A3NFvH).
~~~
@inproceedings{
chu2020multiagent,
title={Multi-agent Reinforcement Learning for Networked System Control},
author={Tianshu Chu and Sandeep Chinchali and Sachin Katti},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=Syx7A3NFvH}
}
~~~


# conda install pytorch torchvision torchaudio pytorch-cuda=11.5 -c pytorch -c nvidia
# cd my_deeprl_network_ori
# conda activate russ
# conda deactivate
# python3 test.py --base-dir training/real/direction_aw001c06/ma2c_nclm/ --port 189 train --config-dir config/config_ma2c_nclm_net.ini
# python3 test.py --base-dir training/real/direction_aw001c06/ma2c_nclm/ --port 189 train --config-dir config/config_ma2c_nclm_net_ten_times.ini
# run the code below
# python3 test.py --base-dir real_a1/ma2c_nclm/ --port 189 train --config-dir config/config_ma2c_nclm_net_ten_times.ini
# python3 test.py --base-dir /eva/ evaluate --evaluation-seeds 15


--- Run on docker instructions ---

以下是**從啟動 Docker 容器到完成 PPO 訓練（使用 `config/config_mappo_0611_noMHA_simplest.ini`）的完整指令流程**，適合直接複製貼上：

---

## ✅ PPO 訓練全流程（含 PyG 支援 + 日誌輸出）

### 🧱 第 1 步：建置 Docker Image（如已建好可略過）

```bash
cd ~/best_environment
docker build -t best_environment:latest .
```

---

### 🚀 第 2 步：啟動背景 Docker 容器

```bash
docker run \
  --gpus all \
  -d \
  --name Trainer_PPO_0611_v2 \
  -v /home/russell512/deeprl_0611_ppo:/workspace/my_deeprl_network \
  best_environment:latest \
  sleep infinity
```

> ✅ *注意容器名稱唯一，若重複請先刪除：`docker rm Trainer_PPO_0611_v2`*

---

### 🔧 第 3 步：進入容器

```bash
docker exec -it Trainer_PPO_0611_v2 /bin/bash
```

---

### 🔩 第 4 步：安裝套件與環境（含 PyG）

```bash
# Python 套件安裝
pip install traci sumolib torch

# 安裝 PyG 相依套件
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-1.9.0+cu111.html

pip install torch-geometric

# 進入工作資料夾
cd /workspace/my_deeprl_network

# 設定 SUMO 路徑（如果需要）
export SUMO_HOME="/root/miniconda/envs/py36/share/sumo"

# 安裝 tmux（如尚未安裝）
apt update && apt install -y tmux
```

---

### 🧠 第 5 步：啟動訓練（使用 tmux）

```bash
tmux new -s training_noMHA_0611
```

tmux 內執行以下訓練指令：

```bash
# 設定輸出資料夾與 log
export BASE_DIR_NAME="real_a1/0611_PPO_noMHA_simplest"
mkdir -p ${BASE_DIR_NAME}/log

# 啟用 GAT 模組（如不使用可省略或設為 0）
export USE_GAT=1

# 執行訓練，log 存檔自動加時間戳記
python3 test.py \
    --base-dir ${BASE_DIR_NAME} \
    --port 206 \
    train \
    --config-dir config/config_mappo_0611_noMHA_simplest.ini \
    > ${BASE_DIR_NAME}/log/training_0611_PPO_noMHA_$(date +%Y%m%d_%H%M%S).log 2>&1
```

---

### 📤 第 6 步：分離 tmux，並退出容器

```bash
# 在 tmux 裡按下：
Ctrl + b，放開後按 d

# 然後輸入：
exit
```

---

### 📴 第 7 步：關閉 SSH，訓練仍會持續在容器+tmux 中執行

---

### 🔁 第 8 步：日後重連查看訓練

```bash
ssh <your-server>
docker exec -it Trainer_PPO_0611_v2 /bin/bash
tmux attach -t training_noMHA_0611
```

---

### 📈 可選：啟動 TensorBoard 觀察訓練 log

```bash
tensorboard --logdir=/home/russell512/deeprl_0611_ppo/real_a1 --port=6010
```

瀏覽器開啟：[http://localhost:6010](http://localhost:6010)

---

### 🧹 查看正在訓練的 Python 程序（確認是否仍在跑）

```bash
ps -eo pid,user,%cpu,%mem,cmd | grep python | grep -v grep
```


