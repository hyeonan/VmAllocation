import gym
from gym import spaces
import numpy as np
import sqlite3
from typing import List
from typing_extensions import Literal


def get_tables_in_sqlite(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        return tables
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return None


def load_data_from_sqlite(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vm")
        vm_data = np.array(cursor.fetchall())
        cursor.execute("SELECT * FROM vmType")
        vm_type_data = np.array(cursor.fetchall())
        conn.close()
        return vm_data[:, 1], vm_type_data[:, 1]
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return None
    

class VmAllocationEnv(gym.Env):
    def __init__(self, history_data):
        super(VmAllocationEnv, self).__init__()

        if history_data is None:
            raise ValueError("History data cannot be None")

        self.history_data = history_data
        self.current_step = 0
        self.max_vms = 10
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.max_vms,))
        self.action_space = spaces.Discrete(2)
        self.current_vms = np.zeros(self.max_vms)

    def step(self, action):
        if action == 1:
            empty_slots = np.where(self.current_vms == 0)[0]
            if len(empty_slots) > 0:
                self.current_vms[empty_slots[0]] = self.history_data[self.current_step]
        elif action == 0:
            filled_slots = np.where(self.current_vms > 0)[0]
            if len(filled_slots) > 0:
                self.current_vms[filled_slots[-1]] = 0

        self.current_step += 1
        reward = self.calculate_reward()
        done = self.current_step >= len(self.history_data)
        return self.get_state(), reward, done, {}

    def calculate_reward(self):
        latency = np.mean(self.current_vms == 0)
        throughput = np.sum(self.current_vms)
        acceptance = np.mean(self.current_vms > 0)
        reward = 0.5 * (1 - latency) + 0.3 * throughput + 0.2 * acceptance
        return reward

    def get_state(self):
        return np.sum(self.current_vms > 0)

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
        self.q_table = np.zeros((n_states, n_actions))

    def get_action(self, state):
        state = int(state)
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = np.random.choice(self.n_actions)
        else:
            action = np.argmax(self.q_table[state])
        return action

    def update_q_table(self, state, action, reward, next_state):
        state, next_state = int(state), int(next_state)
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (
                    reward + self.discount_factor * next_max)
        self.q_table[state, action] = new_value


table_names = get_tables_in_sqlite('packing_trace_zone_a_v1.sqlite')

if table_names is None:
    print("Failed to load table names")
else:
    print("Table names: ", table_names)

    history_data, vm_type_data = load_data_from_sqlite('packing_trace_zone_a_v1.sqlite')
    
    if history_data is None:
        print("Failed to load data")
    else:
        env = VmAllocationEnv(history_data)
        n_states = 10
        n_actions = env.action_space.n
        agent = QLearningAgent(n_states=n_states, n_actions=n_actions)

        for episode in range(100):
            state = env.reset()
            done = False
            while not done:
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.update_q_table(state, action, reward, next_state)
                state = next_state
