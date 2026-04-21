AGENT_TYPE="dqn" # q-learning, sarse, dqn
SAVE_PATH="weights/DQN/2stage.pth" # path to save model policy

python main.py \
    --agent_type "${AGENT_TYPE}" \
    --save_path "${SAVE_PATH}" \
    --is_2stage > logs/dqn_2stage.log