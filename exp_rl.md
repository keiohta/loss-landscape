## 環境構築

### Anaconda

```bash
$ conda create -n torch python=3.8 anaconda
$ conda activate torch
$ conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
```

### torchrl

データを生成しないならいらない

```bash
$ cd workspace/rl
$ git clone git@github.com:keiohta/torchrl.git
$ cd torchrl
$ pip install -e .
$ git checkout save_model
```

other libraries

```bash
# 必要に応じてインストール（記録し忘れ）
```

MuJoCo

```bash
$ pip install gym mujoco_py
```

## Pendulumでデータ生成

```bash
$ python examples/run_sac.py --env-name Pendulum-v0 --n-layer 2 --n-unit 256
$ python examples/run_sac.py --env-name Pendulum-v0 --n-layer 16 --n-unit 256
$ python examples/run_sac.py --env-name Pendulum-v0 --n-layer 2 --n-unit 2048
```

データは以下のように配置するとする

```bash
$ $ tree -L 3
.
├── LICENSE
...
├── data
│   ├── Pendulum-v0
│   │   ├── SAC_L16-U256
│   │   ├── SAC_L2-U2048
│   │   └── SAC_L2-U256
```

フォルダの命名規則は、`{RL}_L{n-layer}-U{n-unit}` となっていて、それぞれのフォルダに以下の内容が入っている。

```bash
$ ls data/Pendulum-v0/SAC_L2-U256
20210128T104309.122712.log    critic_q_0040000_L2-U256.pth  data_100
all_transition_L2-U256.pkl    critic_q_0050000_L2-U256.pth  environ.txt
args.txt                      critic_q_0060000_L2-U256.pth  git-diff.txt
command.txt                   critic_q_0070000_L2-U256.pth  git-head.txt
critic_q_0010000_L2-U256.pth  critic_q_0080000_L2-U256.pth  git-log.txt
critic_q_0020000_L2-U256.pth  critic_q_0090000_L2-U256.pth  git-status.txt
critic_q_0030000_L2-U256.pth  critic_q_0100000_L2-U256.pth
```

- `critic_q_{steps}_L{n-layer}-U{n-unit}.pth`: `steps` 数学習した時のモデル。基本的に最新（10万ステップ学習）を使う
- `all_transition_L{n-layer}-U{n-unit}.pkl`: 学習中に保存したデータが格納されているファイル

## 可視化

2Dのコンター図の生成

```
mpirun -n 4 python plot_surface.py --mpi --cuda --exp-name L2-U256 --model custom --x=-1:1:51 --y=-1:1:51 --dir_type weights --xnorm filter --xig biasbn --ynorm filter --yignore biasbn  --plot
```