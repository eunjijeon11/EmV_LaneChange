import gymnasium as gym
import numpy as np
import traci
from gymnasium import spaces
import agents.emv as emv_logic

EMV_ID = "ambulance"
RADIUS = 80  # meters

# ==============================
# Utils
# ==============================

def get_relative_position(vid, emv_id, all_vehicles):
    """캐시된 차량 정보로 상대 위치 계산"""
    try:
        my_pos    = all_vehicles[vid]['pos']
        emv_pos   = all_vehicles[emv_id]['pos']
        rel_dist  = emv_pos - my_pos  # 음수: 구급차가 뒤에서 접근
        emv_lane  = all_vehicles[emv_id]['lane']
        emv_speed = all_vehicles[emv_id]['speed']
        return (rel_dist, 0), emv_lane, emv_speed
    except:
        return (0, 0), 0, 0

# ==============================
# Environment
# ==============================
class V2XEnv(gym.Env):
    def __init__(self, emv_mode: bool, max_vehicles: int = 36):
        super().__init__()
        self.emv_mode     = emv_mode
        self.max_vehicles = max_vehicles  

        # 5 base + 4 EMV info = 9
        self.observation_space = spaces.Box(
            low=-200, high=200, shape=(9,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        self.prev_lane              = {}
        self.step_count             = 0
        self.collision_recorded     = set()
        self.emv_collision_recorded = set()

    def reset(self, sumo_cmd):
        self.prev_lane              = {}
        self.step_count             = 0
        self.collision_recorded     = set()
        self.emv_collision_recorded = set()
        self._vehicle_cache         = {} 

        try:
            traci.close()
        except:
            pass

        traci.start(sumo_cmd)
        self._vehicle_cache = self._cache_vehicles()  
        return self._get_obs()

    def step(self, actions):
        veh_ids = list(traci.vehicle.getIDList()) 
        info    = {"ov_lanechange": 0, "ov_collisions": 0}

        if self.emv_mode and EMV_ID in veh_ids:
            emv_action = emv_logic.get_emv_action(EMV_ID)
            self._apply_action(EMV_ID, emv_action)

            depart = traci.vehicle.getDeparture(EMV_ID)
            now    = traci.simulation.getTime()
            info["emv_drivetime"] = (now - depart) if depart >= 0 else 0 
            info["emv_speed"]     = traci.vehicle.getSpeed(EMV_ID)

        ov_ids = [v for v in veh_ids if v != EMV_ID]
        for i, vid in enumerate(ov_ids):
            if i >= len(actions):
                break
            applied = self._apply_action(vid, actions[i])  
            if actions[i] in [1, 2] and applied:
                info["ov_lanechange"] += 1
        info["ov_lanechange"] /= len(ov_ids) if len(ov_ids) else 1

        traci.simulationStep()
        self.step_count += 1

        collisions = traci.simulation.getCollisions()
        info["ov_collisions"] = len(collisions) / len(ov_ids) if len(ov_ids) else 0
        
        self._vehicle_cache = self._cache_vehicles()  

        obs     = self._get_obs()
        rewards = self._get_rewards()

        done = (traci.simulation.getMinExpectedNumber() == 0) 

        return obs, rewards if len(rewards) else [0], done, info

    # ==========================
    # Observation
    # ==========================
    def _get_obs(self):
        all_vehicles = self._vehicle_cache
        obs_list     = []

        for vid, info in all_vehicles.items():
            if self.emv_mode and vid == EMV_ID:
                continue

            speed = info['speed']
            lane  = info['lane']

            leader = info['leader']  
            gap    = leader[1] if leader else 100.0
            gap    = min(gap, 100.0)

            left_clear  = self._lane_clear_cached(vid, -1, all_vehicles)
            right_clear = self._lane_clear_cached(vid, +1, all_vehicles)

            base_obs = [lane, speed, gap, left_clear, right_clear]

            if self.emv_mode and EMV_ID in all_vehicles:
                rel_pos, emv_lane, emv_speed = get_relative_position(vid, EMV_ID, all_vehicles)
                emv_obs = [rel_pos[0], rel_pos[1], emv_lane, emv_speed]
            else:
                emv_obs = [0.0, 0.0, -1.0, -1.0]

            obs_list.append(base_obs + emv_obs)

        while len(obs_list) < self.max_vehicles:
            obs_list.append([0.0] * 9)

        return np.array(obs_list[:self.max_vehicles], dtype=np.float32)

    # ==========================
    # Action
    # ==========================
    def _apply_action(self, vid, action):
        try:
            lane       = traci.vehicle.getLaneIndex(vid)
            curr_speed = traci.vehicle.getSpeed(vid)

            if action == 1 and lane > 0:       # LEFT
                traci.vehicle.changeLaneRelative(vid, -1, 1)
            elif action == 2 and lane < 2:     # RIGHT
                traci.vehicle.changeLaneRelative(vid, +1, 1)
            elif action == 3:                  # ACCEL
                traci.vehicle.slowDown(vid, min(curr_speed + 5, 20), 1)
            elif action == 4:                  # DECEL
                traci.vehicle.slowDown(vid, max(0, curr_speed - 5), 1)
            else:                              # KEEP (action == 0)
                return True
            return True
        except Exception as e:
            print(f"[apply_action error] {vid}: {e}")
            return False

    def _lane_clear_cached(self, vid, direction, all_vehicles):
        try:
            my_lane     = all_vehicles[vid]['lane']
            my_pos      = all_vehicles[vid]['pos']
            target_lane = my_lane + direction
            if target_lane < 0 or target_lane > 2:
                return 0
            for other, info in all_vehicles.items():
                if other == vid:
                    continue
                if info['lane'] == target_lane and abs(info['pos'] - my_pos) < 15:
                    return 0
            return 1
        except:
            return 0

    def _cache_vehicles(self):
        cache = {}
        for vid in traci.vehicle.getIDList():
            try:
                cache[vid] = {
                    'lane':    traci.vehicle.getLaneIndex(vid),
                    'pos':     traci.vehicle.getLanePosition(vid),
                    'speed':   traci.vehicle.getSpeed(vid),
                    'lane_id': traci.vehicle.getLaneID(vid),
                    'leader':  traci.vehicle.getLeader(vid),  #캐싱
                }
            except:
                pass
        return cache

    # ==========================
    # Reward
    # ==========================
    def _get_rewards(self):
        rewards = []

        # 공통 데이터 호출
        all_vehicles = self._vehicle_cache 
        colliding    = set(traci.simulation.getCollidingVehiclesIDList())
        collisions   = traci.simulation.getCollisions()

        # EMV 충돌 차량 사전 추출 
        emv_colliders = set()
        for c in collisions:
            if c.collider == EMV_ID: emv_colliders.add(c.victim)
            if c.victim   == EMV_ID: emv_colliders.add(c.collider)

        emv_exists = self.emv_mode and EMV_ID in all_vehicles
        if emv_exists:
            emv_lane  = all_vehicles[EMV_ID]['lane']
            emv_pos   = all_vehicles[EMV_ID]['pos']
            emv_speed = all_vehicles[EMV_ID]['speed']

        for vid, vinfo in all_vehicles.items():
            if vid == EMV_ID:
                continue

            speed     = vinfo['speed']
            curr_lane = vinfo['lane']
            veh_pos   = vinfo['pos']

            try:
                max_speed = traci.lane.getMaxSpeed(vinfo['lane_id'])
            except:
                max_speed = 20.0

            # --- B. 일반 주행 보상 ---
            # 속도 보상 스케일 +2.0
            r  =  2.0 * (speed / max_speed)
            # 속도 오차 벌점 
            r -= 0.01 * ((max_speed - speed) ** 2)

            # 정지 벌점
            if speed < 1.0:
                r -= 10.0

            # 일반 충돌 벌점 (1회만)
            if vid in colliding and vid not in self.collision_recorded:
                r -= 100.0
                self.collision_recorded.add(vid)
                
            # EMV 회피 차선 변경은 벌점 면제
            emv_evasion = False
            if emv_exists:
                _dist = emv_pos - veh_pos
                prev_lane_vid = self.prev_lane.get(vid, curr_lane) 

                emv_evasion = (
                    prev_lane_vid  == emv_lane and
                    curr_lane != emv_lane and
                    -80 < _dist < 0
                )

            # 차선 변경 벌점 -0.3
            prev_lane_vid = self.prev_lane.get(vid, curr_lane)
            if vid in self.prev_lane and self.prev_lane[vid] != curr_lane:
                if not emv_evasion:  # [수정] 회피 차선 변경은 벌점 면제
                    r -= 3.0
            self.prev_lane[vid] = curr_lane

            # --- C.구급차 대응 보상 ---
            if emv_exists:
                dist_to_emv = emv_pos - veh_pos  # 음수: 구급차가 뒤에서 접근
                same_lane   = (curr_lane == emv_lane)

                # 1. 구급차 80m 이내 후방 접근
                if same_lane:
                    # 거리 비례 벌점 (매 스텝 누적, 스케일 조정)
                    distance_penalty = min(10.0 * (80.0 / (abs(dist_to_emv) + 1.0)), 20.0)
                    r -= distance_penalty
                else:
                    # 유지 보상 +5
                    r += 5.0
                    # 회피 순간 보상 +30
                    if prev_lane_vid == emv_lane and curr_lane != emv_lane:
                        r += 30.0

                # 2.구급차 옆 차선 근접 시 감속 협조
                if not same_lane and -30 < dist_to_emv < 0:
                    if speed < emv_speed * 0.8:
                        r += 3.0
                    elif speed > emv_speed:
                        r -= 2.0

                # 3.구급차 직접 충돌 (1회만)
                if vid in emv_colliders and vid not in self.emv_collision_recorded:
                    r -= 50.0
                    self.emv_collision_recorded.add(vid)

            rewards.append(r)

        # max_vehicles 패딩
        while len(rewards) < self.max_vehicles:
            rewards.append(0.0)

        return rewards

    def close(self):
        try:
            traci.close()
        except:
            pass