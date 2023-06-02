import gym
from gym import spaces
import numpy as np
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim


def load_vm_requests(db_path, limit=1000):  # new 'limit' parameter
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT vmId, vmTypeId, starttime, endtime FROM vm LIMIT ?", (limit,))  # limit the number of rows
        vm_requests = np.array(cursor.fetchall())
        conn.close()
        return vm_requests
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return None


def load_vm_types(db_path, limit=1000):  # new 'limit' parameter
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, core, memory FROM vmType LIMIT ?", (limit,))  # limit the number of rows
        vm_types = np.array(cursor.fetchall())
        conn.close()
        return vm_types
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return None



class QNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(n_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VmAllocationEnv(gym.Env):
    def __init__(self, vm_requests, vm_types):
        super(VmAllocationEnv, self).__init__()

        if vm_requests is None or vm_types is None:
            raise ValueError("VM requests and VM types data cannot be None")

        self.vm_requests = vm_requests
        self.vm_types = vm_types
        self.current_step = 0
        self.max_vms = np.max(vm_requests[:, 0])  # Maximum number of virtual machines
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.max_vms,))
        self.action_space = spaces.Discrete(2)
        self.current_vms = np.zeros(self.max_vms)

    def step(self, action):
        if action == 1:
            current_request = self.vm_requests[self.current_step]
            vm_type_id = current_request[1]  # VM type ID of the request
            starttime = current_request[2]  # Start time of the request
            endtime = current_request[3]  # End time of the request

            num_vms = 0
            if starttime is not None and endtime is not None:
                num_vms = int(endtime - starttime)  # Number of VMs requested

            empty_slots = np.where(self.current_vms == 0)[0]
            if len(empty_slots) >= num_vms:
                assigned_vms = np.random.choice(empty_slots, size=num_vms, replace=False)
                for vm_idx in assigned_vms:
                    self.current_vms[vm_idx] = vm_type_id
                    print(f"Assigned VM {vm_idx} with VM type {vm_type_id}")
        elif action == 0:
            filled_slots = np.where(self.current_vms > 0)[0]
            if len(filled_slots) > 0:
                prev_vm_idx = filled_slots[-1]
                self.current_vms[prev_vm_idx] = 0
                print(f"Removed VM {prev_vm_idx}")

        self.current_step += 1
        reward = self.calculate_reward()
        done = self.current_step >= len(self.vm_requests)
        return self.get_state(), reward, done, {}

    def calculate_reward(self):
        total_vms = len(self.current_vms)
        occupied_vms = np.sum(self.current_vms > 0)
        uniform_allocation = 1 - np.abs(occupied_vms / total_vms - 0.5)  # Uniform allocation reward
        reward = uniform_allocation
        return reward

    def calculate_occupancy_rate(self):
        # Calculate occupancy rate
        total_vms = len(self.current_vms)
        occupied_vms = np.sum(self.current_vms > 0)
        occupancy_rate = occupied_vms / total_vms
        return occupancy_rate

    def get_state(self):
        return self.current_vms

    def reset(self):
        self.current_step = 0
        self.current_vms = np.zeros(self.max_vms)
        return self.get_state()


class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.01, discount_factor=0.99, exploration_rate=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        self.q_network = QNetwork(n_states, n_actions)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = np.random.choice(self.n_actions)
            print(f"Exploration: Selected random action {action}")
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                action = torch.argmax(q_values, dim=1).item()
                print(f"Exploitation: Selected action {action} with highest Q-value")
        return action

    def update_q_network(self, state, action, reward, next_state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        q_values = self.q_network(state)
        q_value = q_values[0, action]
        with torch.no_grad():
            next_q_values = self.q_network(next_state)
            next_max_q_value = torch.max(next_q_values)
        target_q_value = reward + self.discount_factor * next_max_q_value
        loss = nn.MSELoss()(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


db_path = 'packing_trace_zone_a_v1.sqlite'  # Dataset file path
vm_requests = load_vm_requests(db_path)
vm_types = load_vm_types(db_path)

if vm_requests is None or vm_types is None:
    print("Failed to load data")
else:
    env = VmAllocationEnv(vm_requests, vm_types)
    n_states = env.max_vms
    n_actions = env.action_space.n
    agent = QLearningAgent(n_states=n_states, n_actions=n_actions)

    num_episodes = 100
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        print(f"Episode: {episode + 1}")
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_network(state, action, reward, next_state)
            state = next_state
            total_reward += reward


                # Calculate and save occupancy rate at the end of the episode
        occupancy_rate = env.calculate_occupancy_rate()
        with open(f"Occupancy_rate_episode_{episode+1}.txt", "w") as f:
            f.write(f"{occupancy_rate:.2f}\n")   

        print(f"Total Reward: {total_reward:.2f}")
        print("———————————")


