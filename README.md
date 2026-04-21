# 🚑 EMV-Aware Lane Changing under Real-World Perception Constraints with Two-Stage Learning
Eunji Jeon, Dain Kim, Meng Xu, Seonwoo Park  
Sungkyunkwan University

## Abstract & Overview

## ⚙️ Environment Setup
First, download the [SUMO-RL](https://sumo.dlr.de/docs/Downloads.php#sumo_-_latest_release_version_1260) simulator. It differs by your OS.  
Below is the download example for Linux (Ubuntu).
```
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

export SUMO_HOME="/usr/share/sumo"
export LIBSUMO_AS_TRACI=1
```
Next, download the python libraries in [requirements.txt](./requirements.txt).
```
conda create --name=emv python=3.10
conda activate emv
pip install -r requirements.txt
```
Running the DQN model may require additional CUDA configuration.

## 👩🏻‍🏫 Train and Evaluation
To run train & evaluation code, revise [run_experiment.sh](./run_experiment.sh) file and run it.
```
bash run_experiment.sh
```
Or, directly run the [main.py](./main.py) code.
```
main.py --agent_type dqn --save_path weights/DQN/2stage.pth --is_2stage
```

## 📁 Project Structure
```
📁 EmV_LaneChange/
├── 📁 agents/
│ ├── base.py # BaseAgent class
│ ├── emv.py # EmV Dijkstra implementation
│ ├── dqnAgent.py # DQN implementation
│ ├── sarsaAgent.py # Sarsa implementation
│ └── qAgent.py # Q-learning implementation
│
├── 📁 configs/
│ ├── net.xml # Road configuration
│ ├── route.xml # Vehicle configuration (randomly generated during inference)
│ └── emvconfig.sumocfg # sumo configuration
│
├── 📁 weights/
│ ├── 📁 DQN
│ │ ├── 2stage.pth
│ │ └── single_stage.pth
│ ├── 📁 Q-Learning
│ │ ├── 2stage.pkl
│ │ └── single_stage.pkl
│ └── 📁 SARSA
│   ├── 2stage.pkl
│   └── single_stage.pkl
│
├── environment.py # rewards, observation and actions
├── main.py # Train & Evaluation
├── run_experiments.sh # Train & Evaluation
├── requirements.txt
└── README.md
```
