from environment_v2x import V2XEnv
from agents import QAgent, SarsaAgent, DQNAgent
import random
import argparse
import os

SUMO_CMD = ["sumo", "-c", "configs/emvconfig.sumocfg"]

def generate_random_route(emv_mode, vehicle_num):
    with open("configs/route.xml", 'w') as f:
        print("""<?xml version="1.0" encoding="UTF-8"?>
        <routes>
            <vType id="car" accel="2.5" decel="4.5" sigma="0.3" length="5" maxSpeed="20"/>
            <vType id="emv" accel="2.5" decel="4.5" sigma="0.3" length="8" maxSpeed="25"/>

            <route id="r0" edges="road0"/>\n""", file=f)
        
        lanes = [0, 1, 2]
        vehicles = []

        for i in range(vehicle_num):
            depart = random.randint(0, 400) / 10.0
            vehicles.append(["ov", depart])

        if emv_mode:
            depart = random.randint(100, 400) / 10.0
            vehicles.append(["emv", depart])
        
        # depart 기준 정렬
        vehicles.sort(key=lambda x: x[1])

        # 출력
        for vid, vinfo in enumerate(vehicles):
            if vinfo[0] == "emv":
                print(f'\t\t<vehicle id="ambulance" type="emv" route="r0" depart="{vinfo[1]}" departLane="random" departSpeed="random" departPos="base"/>\n', file=f)
            else:
                print(f'\t\t<vehicle id="veh{vid}" type="car" route="r0" depart="{vinfo[1]}" departLane="random" departSpeed="random" departPos="base"/>\n', file=f)
        
        print("</routes>", file=f)

def get_agent(agent_type):
    if agent_type == "q-learning":
        return QAgent
    elif agent_type == "sarsa":
        return SarsaAgent
    elif agent_type == "dqn":
        return DQNAgent
    else:
        raise NotImplementedError

def train(model, env, emv_mode, args):
    for eps in range(args.train_episode_num):
        obs = env.reset(SUMO_CMD)        
        generate_random_route(emv_mode, args.vehicle_num)
        total_reward = 0
        step_count = 0
    
        while True:
            actions = model.predict(obs)
            next_obs, reward, done, info = env.step(actions)
            
            model.update(obs, actions, reward, next_obs, done)
            obs = next_obs
            
            if isinstance(reward, list):
                total_reward += sum(reward) / len(reward) if len(reward) > 0 else 0
            else:
                total_reward += reward
            step_count += 1

            if done:
                env.close()
                break

        if eps % 10 == 0:
            model.episode_done()
        
        if model.epsilon > model.epsilon_min:
            model.epsilon *= model.epsilon_decay

        print(f"Episode {eps} | Reward: {total_reward:.2f} | steps: {step_count} | epsilon: {model.epsilon:.4f}")
    
    return model

# ==============================
# Config
# ==============================

parser = argparse.ArgumentParser()

parser.add_argument("--agent_type", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)

parser.add_argument("--train_episode_num", type=int, default=200)
parser.add_argument("--test_episode_num", type=int, default=10)
parser.add_argument("--vehicle_num", type=int, default=30)

parser.add_argument("--is_2stage", action="store_true")

args = parser.parse_args()

# ==============================
# Training
# ==============================

base_env = V2XEnv(emv_mode=False)
model = get_agent(args.agent_type)(base_env.action_space)

if args.is_2stage:
    model = train(model, base_env, False, args)
    print("Base OV training done!")

emv_env = V2XEnv(emv_mode=True)
model = train(model, emv_env, True, args)
print("EmV aware LC training done!")

os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
model.save_policy(args.save_path)

# ==============================
# Inference
# ==============================
test_env = V2XEnv(emv_mode=True)
model = get_agent(args.agent_type)(test_env.action_space)
model.load_policy(policy_dir=args.save_path)

emv_passage_time_sum = 0
emv_speed_sum = 0
ov_lanechange_sum = 0
ov_collision_sum = 0

for eps in range(args.test_episode_num):
    emv_passage_time = -1
    emv_speed_history = []
    ov_lanechange_count = 0
    ov_collision_count = 0
    
    obs = test_env.reset(SUMO_CMD)
    generate_random_route(True, args.vehicle_num)
    
    while True:
        actions = model.predict(obs)
        next_obs, reward, done, info = test_env.step(actions)
        
        if "emv_drivetime" in info:
            emv_passage_time = info["emv_drivetime"]
            emv_speed_history.append(info["emv_speed"])
        ov_lanechange_count += info["ov_lanechange"]
        ov_collision_count += info["ov_collisions"]
        
        obs = next_obs
        
        if done:
            test_env.close()
            break

    print(f"""Episode: {eps}
          EmV passage time: {emv_passage_time}
          EmV average_speed: {sum(emv_speed_history)/len(emv_speed_history)}
          OV lane change: {ov_lanechange_count}
          OV collision: {ov_collision_count}\n\n""")
    
    emv_passage_time_sum += emv_passage_time
    emv_speed_sum += sum(emv_speed_history)/len(emv_speed_history)
    ov_lanechange_sum += ov_lanechange_count
    ov_collision_sum += ov_collision_count

print(f"""======================Average over {args.test_episode_num} episodes======================
          EmV passage time: {emv_passage_time_sum/args.test_episode_num}
          EmV average_speed: {emv_speed_sum/args.test_episode_num}
          OV lane change: {ov_lanechange_sum/args.test_episode_num}
          OV collision: {ov_collision_sum/args.test_episode_num}\n\n""")