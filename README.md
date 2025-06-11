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

以下是針對您新的實驗資料夾與設定檔 `/home/russell512/deeprl_0611_ppo/config/config_mappo_nc_net_exp_0519.ini` 的完整操作指引，已將相關路徑與容器名稱更新：

---

## ✅ **完全 WORK for PPO 實驗 - deeprl\_0611\_ppo**

### 🎯目標：使用 Docker + tmux 讓訓練在背景穩定持續執行，不受 SSH 斷線影響

---

## **更新參數說明**

* **專案資料夾**：`/home/russell512/deeprl_0611_ppo`
* **config 檔案**：`config/config_mappo_nc_net_exp_0519.ini`
* **容器名稱建議**：`Trainer_PPO_0611`
* **TensorBoard logdir**：`/home/russell512/deeprl_0611_ppo/real_a1`
* **Port 建議**：199（避免與其他實驗衝突）

---

### 🧱 **第 1 步：建置 Docker Image**

```bash
cd ~/best_environment
docker build -t best_environment:latest .
```

---

### 🚀 **第 2 步：啟動背景容器**

```bash
docker run \
  --gpus all \
  -d \
  --name Trainer_PPO_0611 \
  -v /home/russell512/deeprl_0611_ppo:/workspace/my_deeprl_network \
  best_environment:latest \
  sleep infinity
```

---

### 🔧 **第 3 步：進入容器**

```bash
docker exec -it Trainer_PPO_0611 /bin/bash
```

---

### 🔩 **第 4 步：環境設置**

```bash
pip install traci sumolib torch

cd /workspace/my_deeprl_network

export SUMO_HOME="/root/miniconda/envs/py36/share/sumo"

apt update
apt install -y tmux
```

---

### 🧠 **第 5 步：啟動訓練 (tmux)**

```bash
tmux new -s training_ppo_0611
```

接著在 tmux 中執行：

```bash
mkdir -p real_a1/log

export USE_GAT=1
python3 test.py \
  --base-dir real_a1 \
  --port 199 \
  train \
  --config-dir config/config_mappo_nc_net_exp_0519.ini \
  > real_a1/log/training_ppo_0611_$(date +%Y%m%d_%H%M%S).log 2>&1
```

---

### 🔚 **第 6、7 步：分離並退出容器**

```bash
# 在 tmux 裡按：Ctrl + b，然後按 d 鍵
exit  # 回到主機
```

---

### 📶 **第 8 步：安全地斷線 SSH**

您現在可以放心中斷 SSH 連線了，訓練仍會在容器 + tmux 中持續執行。

---

### 🔁 **第 9 步：重新連線查看訓練**

```bash
ssh <your-server>

docker exec -it Trainer_PPO_0611 /bin/bash

tmux attach -t training_ppo_0611
```

---

### 📈 **啟動 TensorBoard 監控**

```bash
tensorboard --logdir=/home/russell512/deeprl_0611_ppo/real_a1 --port=6009
```

然後瀏覽器開啟：
[http://localhost:6009](http://localhost:6009)

---

### 🧹 **監控訓練程序**

```bash
ps -eo pid,user,%cpu,%mem,cmd | grep python | grep -v grep
```


