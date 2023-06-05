import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sqlite3
import sys
from tqdm.notebook import tqdm

import torch
from torch import optim, nn
import torch.nn.functional as F
import gym
from gym import spaces

# hyperparameters
hidden_size1 = 512
hidden_size2 = 256
actor_learning_rate = 1e-5
critic_learning_rate = 1e-5

# Constants
GAMMA = 0.97
waiting_penalty_weight = 0.05
num_steps = 200
max_episodes = 100

def load_vm_reqests(path, n_data=10000):
  con = sqlite3.connect(path)
  cur = con.cursor()
  cur.execute('SELECT * FROM vm WHERE starttime >= 0 AND priority = 1 ORDER BY starttime asc Limit ?', (n_data,))
  vm_requests = cur.fetchall()
  names = list(map(lambda x: x[0], cur.description))
  vm_df = pd.DataFrame(data=vm_requests, columns=names)
  vm_df = vm_df.sort_values(by="starttime").reset_index(drop=True)
  vm_df["starttime"] = vm_df["starttime"].apply(lambda x : round(x,5))
  vm_df["endtime"] = vm_df["endtime"].apply(lambda x : round(x,5))
  vm_df["endtime"] = vm_df["endtime"].fillna(np.inf)

  return vm_df

def load_vm_type(path):
  con = sqlite3.connect(path)
  cur = con.cursor()
  cur.execute('SELECT * FROM vmType')
  vmType = cur.fetchall()
  names = list(map(lambda x: x[0], cur.description))
  vmType_df = pd.DataFrame(data=vmType, columns=names)
  vmType_df["core"].apply(lambda x : round(x,6))
  return vmType_df
  
def access_cpu_usage(request_table, type_table, vm_id, machine_id):
  type_id = request_table[request_table["vmId"] == vm_id]["vmTypeId"].values[0]
  cur_type_table = type_table[type_table["vmTypeId"]==type_id]
  cpu_usage = cur_type_table[cur_type_table["machineId"] == machine_id]["core"].values[0]
  return cpu_usage

def update_state_requests(request_table, type_table, time_step, n_machines, state):
	next_request_id = request_table.loc[time_step, "vmId"]
	next_request_type_id = request_table.loc[time_step, "vmTypeId"]
	machine_id_list = type_table[type_table["vmTypeId"]==next_request_type_id]["machineId"].values
	for machine_id in machine_id_list:
			next_cpu_usage = access_cpu_usage(request_table, type_table, next_request_id, machine_id)
			state[n_machines+machine_id] = next_cpu_usage		

class VmAllocEnv(gym.Env):
  def __init__(self, vm_requests, vm_types):
    self.n_machines = vm_types["machineId"].nunique()
    '''
    state[:self.n_machines] : CPU usage of each machine
    state[self.n_machines:] : requested cpu amount for each machine 
    from current request's type
    '''
    self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_machines*2,), dtype=np.float32)
    self.action_space = spaces.Discrete(self.n_machines)
    self.vm_requests = vm_requests
    self.vm_types = vm_types
    self.cur_step = 0
    self.state = np.zeros(self.n_machines*2)
    self.state[self.n_machines:]=1
    # {VM Id : Machine Id)
    self.allocated_vm = {}

  def remove_expired_req(self, time):
    # Remove expired VM requests from current state
    remove_table = self.vm_requests.loc[self.allocated_vm.keys(), "endtime"] < time
    remove_list = remove_table[remove_table].index
    if len(remove_list)>0:
      for vm_ind in remove_list:
        alloc_machine_id = self.allocated_vm[vm_ind]
        vm_id = self.vm_requests.loc[vm_ind, "vmId"]
        cpu_usage = access_cpu_usage(self.vm_requests, self.vm_types, vm_id, alloc_machine_id )
        # print("Before decrease : ", self.state[alloc_machine_id])
        self.state[alloc_machine_id] = self.state[alloc_machine_id] - cpu_usage
        # print("After decrease", self.state[alloc_machine_id])
        del self.allocated_vm[vm_ind]

  def step(self, action):
    # Update state : Increase CPU usuage of current state by action
    cur_vm_id = self.vm_requests.loc[self.cur_step, "vmId"]
    cur_cpu_usage = access_cpu_usage(self.vm_requests, self.vm_types, cur_vm_id, action)
    self.state[action] = self.state[action] + cur_cpu_usage
    self.allocated_vm[self.cur_step] = action

    # Uodate state : Increase time step, Get next request and update current state
    self.cur_step += 1
    self.state[self.n_machines:] = 1
    update_state_requests(self.vm_requests, self.vm_types, self.cur_step, 
    self.n_machines, self.state)
    
    # Calculate reward
    reward_vals = self.state[:self.n_machines]
    # variance = np.sum(np.square(reward_vals/reward_vals.mean() - 1))
    reward = 1 - reward_vals.std()
    
    # Check done
    done = self.cur_step >= len(self.vm_requests)-1

    return self.state, reward, done, {}

  def reset(self):
      self.cur_step = 0
      self.allocated_vm = {}
      self.state = np.zeros(self.n_machines*2)
      self.state[self.n_machines:]=1
      update_state_requests(self.vm_requests, self.vm_types, self.cur_step, 
      self.n_machines, self.state)
      return self.state

class Critic(nn.Module):
    def __init__(self, num_inputs, hidden_size_1, hidden_size_2):
        super(Critic, self).__init__()

        self.critic_linear1 = nn.Linear(num_inputs, hidden_size_1)
        self.critic_linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.critic_linear3 = nn.Linear(hidden_size_2, 1)

        # for m in self.modules():
        #   if isinstance(m, nn.Linear):
        #     nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        #     nn.init.constant_(m.bias.data, 0)
    
    def forward(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        value = F.relu(self.critic_linear1(state))
        value = F.relu(self.critic_linear2(value))
        value = self.critic_linear3(value)
      
        return value

class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size_1, hidden_size_2):
        super(Actor, self).__init__()

        self.num_actions = num_actions

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size_1)
        self.actor_linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.actor_linear3 = nn.Linear(hidden_size_2, num_actions)

        # for m in self.modules():
        #   if isinstance(m, nn.Linear):
        #     nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        #     nn.init.constant_(m.bias.data, 0)

    
    def forward(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.relu(self.actor_linear2(policy_dist))
        policy_dist = F.softmax(self.actor_linear3(policy_dist), dim=1)

        return policy_dist

def main():
    path = './packing_trace_zone_a_v1.sqlite'
    # Load dataz
    vm_df = load_vm_reqests(path, num_steps)
    vmType_df = load_vm_type(path)
    # Setup environment
    env = VmAllocEnv(vm_df, vmType_df)

    # Model configuration
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n

    critic = Critic(num_inputs, hidden_size1, hidden_size2)
    actor = Actor(num_inputs, num_outputs, hidden_size1, hidden_size2)
    critic_optimizer  = optim.Adam(critic.parameters(), lr=actor_learning_rate)
    actor_optimizer = optim.Adam(actor.parameters(), lr=critic_learning_rate)

    # Log data
    all_lengths = []
    average_lengths = []
    all_rewards = []
    cpu_usage_list = [[]]
    all_halt_resume = []

    time_interval = 0.00011
    train_iterator = tqdm(range(max_episodes), desc="Episode",mininterval=10, ncols=100)
    for episode in train_iterator:
      probs = []
      values = []
      rewards = []
      halt_resume = []
      # Reset environment
      state = env.reset()
      done = False
      steps = 0
      waiting_penalty = 0

      cur_sign = 1
      time_clock = 0
      while(done != True):
        cur_state_starttime = env.vm_requests.loc[env.cur_step, "starttime"]
        if (time_clock <= cur_state_starttime) & (time_clock + time_interval > cur_state_starttime):
          env.remove_expired_req(time_clock)
          prev_sign = cur_sign
          # Get value, action distribution from ActorCritic funtion
          torch.autograd.set_detect_anomaly(True)
          value = critic.forward(state)
          policy_dist = actor.forward(state)
          value = value[0,0]
          dist = policy_dist.detach().numpy() 
          # Mask not supporting machine for such vm type
          type_mask = (state[env.n_machines:] < 1).astype("int")
          # Mask cpu overload machine
          check_overload = state[:env.n_machines] + state[env.n_machines:]
          overload_mask = (check_overload <= 1).astype("int")
          action_prob = dist * type_mask * overload_mask
          # Failed to allocate all the VMs. Wait until appropriate machine resources are available.
          if action_prob.sum() == 0:
            waiting_penalty += 1
            steps += 1
            time_clock += time_interval
            env.vm_requests.loc[env.cur_step:,"starttime"] += time_interval
            env.vm_requests.loc[env.cur_step:,"endtime"] += time_interval
            cur_sign = 0
          else:
            action_prob = action_prob/(action_prob.sum())
            # Choose action from given distribution
            action = np.random.choice(num_outputs, p=np.squeeze(action_prob))
            prob = policy_dist.squeeze(0)[action]
            # Take action, go to next step
            new_state, reward, done, _ = env.step(action)
            # Penalize waited steps
            reward = reward + waiting_penalty_weight * (-waiting_penalty)
            rewards.append(reward)
            values.append(value)
            probs.append(prob)
            state = new_state
            steps += 1
            waiting_penalty = 0
            cur_sign = 1
            cpu_usage_list[episode].append(state[:env.n_machines].tolist())
          if (cur_sign==0)&(prev_sign==1):
            halt = steps
          if (cur_sign==1)&(prev_sign==0):
            resume = steps
            halt_resume.append([halt,resume, env.cur_step-1])
        else:
          time_clock += time_interval


      # End of each episode
      all_rewards.append(np.sum(rewards))
      all_lengths.append(steps)
      average_lengths.append(np.mean(all_lengths[-10:]))
      
      values = torch.stack(values)
      # compute Q values
      Qval = 0
      Qvals = torch.zeros_like(values)
      for t in reversed(range(len(rewards))):
          Qval = rewards[t] + GAMMA * Qval
          Qvals[t] = Qval


      # negative_ind = np.argwhere(Qvals < 0).squeeze(0)
      # probs[negative_ind] = 1-probs[negative_ind]
      probs = torch.stack(probs)
      log_probs = torch.log(probs)
      advantage = Qvals - values

      # Compute critic and actor loss
      critic_loss = advantage.pow(2).mean()
      actor_loss = (-log_probs * advantage.detach()).mean()

      critic_optimizer.zero_grad()
      actor_optimizer.zero_grad()

      critic_loss.backward()
      actor_loss.backward()

      critic_optimizer.step()
      actor_optimizer.step()

      all_halt_resume.append(halt_resume)
      cpu_usage_list.append(state[:env.n_machines].tolist())
      train_iterator.set_description(f"episode: {episode} reward: {np.sum(rewards):.4f}")              
      sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
      print("Critic Loss {:.4f}/ Actor Loss {:.4f}:".format(critic_loss.item(), actor_loss.item()))


    # Plot results
    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(all_rewards)
    plt.plot(smoothed_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()
if __name__ == "__main__":
    main()