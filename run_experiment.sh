AGENT_TYPE="q-learning" # q-learning, sarse, dqn
SAVE_PATH="weights/Q-Learning/2stage.pkl" # path to save model policy

python main.py \
    --agent_type "${AGENT_TYPE}" \
    --save_path "${SAVE_PATH}" \
    --is_2stage > logs/q_2stage_V2X_2.log