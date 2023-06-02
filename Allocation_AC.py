import gym
from gym import spaces
import numpy as np
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


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
                    #print(f"Assigned VM {vm_idx} with VM type {vm_type_id}")
        elif action == 0:
            filled_slots = np.where(self.current_vms > 0)[0]
            if len(filled_slots) > 0:
                prev_vm_idx = filled_slots[-1]
                self.current_vms[prev_vm_idx] = 0
                #print(f"Removed VM {prev_vm_idx}")

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

    def get_state(self):
        return self.current_vms

    def reset(self):
        self.current_step = 0
        self.current_vms = np.zeros(self.max_vms)
        return self.get_state()

    def calculate_occupancy_rate(self):
        total_vms = len(self.current_vms)
        occupied_vms = np.sum(self.current_vms > 0)
        occupancy_rate = occupied_vms / total_vms
        return occupancy_rate


class ActorCriticAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.01, discount_factor=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Define actor network
        self.actor = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
        
        # Define critic network
        self.critic = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate)
        
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()
    
    def update_networks(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = torch.LongTensor([action])
        reward = torch.FloatTensor([reward])
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Update critic network
        value = self.critic(state)
        next_value = self.critic(next_state)
        target = reward + self.discount_factor * (1 - done) * next_value
        value_loss = nn.MSELoss()(value, target.detach())
        
        # Update actor network
        action_probs = self.actor(state)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(action)
        actor_loss = -log_probs * (target - value.detach())
        
        # Backpropagation
        self.optimizer.zero_grad()
        (actor_loss + value_loss).backward()
        self.optimizer.step()



db_path = 'packing_trace_zone_a_v1.sqlite'  # Dataset file path

env = None
n_states = None
n_actions = None
agent = None

num_episodes = 1000
for episode in range(num_episodes):
    # Reload the data for each episode
    vm_requests = load_vm_requests(db_path)
    vm_types = load_vm_types(db_path)
    if vm_requests is None or vm_types is None:
        print(f"Failed to load data for episode {episode + 1}")
        continue  # Skip this episode if data loading failed
    env = VmAllocationEnv(vm_requests, vm_types)
    n_states = env.max_vms
    n_actions = env.action_space.n
    agent = ActorCriticAgent(n_states=n_states, n_actions=n_actions)

    state = env.reset()
    done = False
    total_reward = 0
    print(f"Episode: {episode + 1}")
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update_networks(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    print(f"Total Reward: {total_reward:.2f}")
    print("———————————")
    
    # Calculate occupancy rate and write to file
    occupancy_rate = env.calculate_occupancy_rate()
    with open(f"OccupancyRate_Episode{episode + 1}.txt", "w") as file:
        file.write(str(occupancy_rate))
