import traci
import heapq
import numpy as np

ACTIONS = {
    "KEEP": 0,
    "LEFT": 1,
    "RIGHT": 2,
    "ACCEL": 3,
    "DECEL": 4
}

def get_next_state(state, action):
    lane, v = state
    if action == "LEFT":
        lane = max(0, lane - 1)
    elif action == "RIGHT":
        lane = min(2, lane + 1)
    elif action == "ACCEL":
        v = min(25, v + 1) # 최대 속도 제한
    elif action == "DECEL":
        v = max(0, v - 1)
    return (lane, v)

def collision_penalty(state, neighbors):
    lane, v = state
    penalty = 0
    for n in neighbors:
        if n['lane'] == lane:
            dist = n['rel_pos']
            # 전방 충돌 방지
            if 0 <= dist < 15: 
                penalty += 50000
            # 차선 변경 시 측후방 충돌 방지
            elif -10 < dist < 0:
                penalty += 30000
    return penalty

def lane_block_penalty(state, neighbors):
    lane, v = state
    front_dist = float('inf')
    for n in neighbors:
        if n['lane'] == lane and n['rel_pos'] > 0:
            front_dist = min(front_dist, n['rel_pos'])
    
    # 앞차가 가까울수록 차선 변경을 유도
    if front_dist < 15: return 500
    if front_dist < 30: return 100
    return 0

def compute_cost(state, neighbors, action):
    lane, v, pos = state
    target_speed = 25.0
    cost = 0
    # 1. 안전성 
    cost += collision_penalty(state, neighbors)
    
    # 2. 전방 거리 측정 
    front_dist = float('inf')
    for n in neighbors:
        if n['lane'] == lane:
            dist = n['abs_pos'] - pos 
            if dist > 0:
                front_dist = min(front_dist, dist)
                
    # 3. 속도 보상 및 오차 벌점
    # 3-A. 속도 비례 이득 
    cost -= 2.0 * (v / target_speed)
    
    # 3-B. 속도 오차 제곱 벌점 (목표 속도 유도)
    error_weight = 0.5 if front_dist > 50 else 0.05
    cost += error_weight * ((target_speed - v) ** 2)
    
    # 4. 차선 변경 억제
    if action in ["LEFT", "RIGHT"]:
        cost += 200
    return cost

def dijkstra_action(initial_state, neighbors, depth=4):
    pq = [(0, initial_state, [])]
    visited = set()

    while pq:
        cost, state, path = heapq.heappop(pq)
        if len(path) >= depth:
            return path[0]
        
        state_key = (state, len(path))
        if state_key in visited: continue
        visited.add(state_key)

        for action_name in ACTIONS.keys():
            next_state = get_next_state(state, action_name)
            c = compute_cost(next_state, neighbors, action_name)
            heapq.heappush(pq, (cost + c, next_state, path + [action_name]))
    return "KEEP"

def get_emv_action(EMV_ID):
    try:
        lane = traci.vehicle.getLaneIndex(EMV_ID)
        speed = traci.vehicle.getSpeed(EMV_ID)
        emv_pos = traci.vehicle.getLanePosition(EMV_ID)
        # 초기 state에 pos를 추가
        state = (lane, int(speed), emv_pos)
        neighbors = []
        for v in traci.vehicle.getIDList():
            if v == EMV_ID: continue
            if traci.vehicle.getRoadID(v) == traci.vehicle.getRoadID(EMV_ID):
                neighbors.append({
                    "lane": traci.vehicle.getLaneIndex(v),
                    "abs_pos": traci.vehicle.getLanePosition(v), # 절대 위치 저장
                    "speed": traci.vehicle.getSpeed(v)
                })

        action_name = dijkstra_action(state, neighbors)
        return ACTIONS[action_name]

    except Exception as e:
        return ACTIONS["KEEP"]