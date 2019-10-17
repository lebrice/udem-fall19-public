#!/usr/bin/env python
# coding: utf-8

# In[1]:
from __future__ import absolute_import

import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.helpers import launch_env, wrap_env, view_results_ipython, change_exercise, seedall, force_done, evaluate_policy
from utils.helpers import SteeringToWheelVelWrapper, ResizeWrapper, ImgWrapper

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# # Reinforcement Learning Basics
# 
# Reinforcement Learning, as we saw in lecture, is the idea of learning a _policy_ in order to maximize future (potentially discounted) rewards. Our policy, similar to the imitation learning network, maps raw image observations to wheel velocities, and at every timestep, receives a _reward_ from the environment. 
# 
# Rewards can be sparse (`1` if goal or task is completed, `0` otherwise) or dense; in general, dense rewards make it easier to learn policies, but as we'll see later in this exercise, defining the correct dense reward is an engineering challenge on its own.
# 
# Today's reinforcement learning algorithms are often a mix between _value-based_ and _policy-gradient_ algorithms, instances of what is called an _actor-critic_ formulation. Actor-critic methods have had a lot of research done on them in recent years (especially within in the deep reinforcement learning era), and later in this exercise, we shall also rediscover the formulation's original problems and different methods currently used to stabilize learning.
# 
# We begin by defining two networks, an `Actor` and `Critic`; in this exercise, we'll be using a deep RL algorithm titled _Deep Deterministic Policy Gradients_. 

# In[2]:


class Actor(nn.Module):
    def __init__(self, action_dim, max_action):
        super(Actor, self).__init__()

        # TODO: You'll need to change this!
        flat_size = 0

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(.1)

        self.lin1 = nn.Linear(flat_size, 100)
        self.lin2 = nn.Linear(100, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(x)
        x = self.lr(self.lin1(x))

        x = self.lin2(x)
        x = self.max_action * self.tanh(x)
        
        return x
    
class Critic(nn.Module):
    def __init__(self, action_dim, max_action):
        super(Critic, self).__init__()

        # TODO: You'll need to change this!
        flat_size = 0

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(.1)

        self.lin1 = nn.Linear(flat_size + action_dim, 100)
        self.lin2 = nn.Linear(100, action_dim)

        self.max_action = max_action

    def forward(self, obs, action):
        x = self.bn1(self.relu(self.conv1(obs)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(x)
        x = torch.cat([x, action], 1)
        x = self.lr(self.lin1(x))

        x = self.lin2(x)
        x = self.max_action * self.tanh(x)
        
        return x


# ## Reward Engineering
# 
# In this part of the exercise, we will experiment with the reward formulation. Given the same model, we'll see how the effect of various reward functions changes the final policy trained. 
# 
# In the section below, we'll take a look at the reward function implemented in `gym-duckietown` with a slightly modified training loop. Traditionally, we `reset()` the environment to start an episode, and then `step()` the environment forward for a set amount of time, executing a new action. If this sounds a bit odd, especially for roboticists, you're right - in real robotics, most code runs asynchronously. As a result, although `gym-duckietown` runs locally by stopping the environment, the `AIDO` submissions will run asynchronously, executing the same action until a new one is received.

# In[3]:


def updated_reward(env):
    # Compute the collision avoidance penalty
    pos, angle, speed = env.cur_pos, env.cur_angle, env.speed
    col_penalty = env.proximity_penalty2(pos, angle)

    # Get the position relative to the right lane tangent
    try:
        lp = env.get_lane_pos2(pos, angle)
    except NotInLane:
        reward = 40 * col_penalty
    else:

        # Compute the reward
        reward = (
                +1.0 * speed * lp.dot_dir +
                -10 * np.abs(lp.dist) +
                +40 * col_penalty
        )
    return reward


# In[4]:


nepisodes = 3


# In[5]:


local_env = launch_env()
local_env = wrap_env(local_env)
local_env = ResizeWrapper(local_env)
local_env = ImgWrapper(local_env)

for _ in range(nepisodes):
    done = False
    obs = local_env.reset()
    
    while not done:
        obs, r, done, info = local_env.step(np.random.random(2))
        new_r = updated_reward(local_env)
        print(r, new_r)
 


# In[6]:


view_results_ipython(local_env)


# **Question 0: After understanding the above computed reward, experiment with the constants for each component. What type of behavior does the above reward function penalize? Is this good or bad in context of autonomous driving? Name some other issues that can arise with single-objective optimization. In addition, give three sets of constants and explain qualitatively what types of behavior each penalizes or rewards (note, you may want to use a different action policy than random)**. Place the answers to the above in `reinforcement-learning-answers.txt`
# 
# 
# 

# # The Reinforcement Learning Learning Code
# 
# Below we'll see a relatively naive implementation of the actor-critic training loop, which proceeds as follows: the critic is tasked with a supervised learning problem of fitting rewards acquired by the agent. Then, the policy, using policy gradients, maximizes the return according to the critic's estimate, rather than using Monte-Carlo updates.
# 
# Below, we see an implementation of `DDPGAgent`, a class which handles the networks and training loop. 

# In[7]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent(object):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super(DDPGAgent, self).__init__()

        self.actor = Actor(action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=5e-2)
        
        self.critic = CriticCNN(action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-2)

    def predict(self, state):
        assert state.shape[0] == 3
        state = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99):
        for it in range(iterations):

            # Sample replay buffer
            sample = replay_buffer.sample(batch_size, flat=self.flat)
            state = torch.FloatTensor(sample["state"]).to(device)
            action = torch.FloatTensor(sample["action"]).to(device)
            next_state = torch.FloatTensor(sample["next_state"]).to(device)
            done = torch.FloatTensor(1 - sample["done"]).to(device)
            reward = torch.FloatTensor(sample["reward"]).to(device)

            # Compute the target Q value
            target_Q = self.critic(next_state, self.actor(next_state))
            
            # TODO: - no detach is a subtle, but important bug!
            target_Q = reward + (done * discount * target_Q)

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '{}/{}_actor.pth'.format(directory, filename))
        torch.save(self.critic.state_dict(), '{}/{}_critic.pth'.format(directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('{}/{}_actor.pth'.format(directory, filename), map_location=device))
        self.critic.load_state_dict(torch.load('{}/{}_critic.pth'.format(directory, filename), map_location=device))


# You'll notice that the training loop needs a `replay_buffer` object. In value-based and actor-critic methods in deep reinforcement learning, the use of a replay buffer is crucial. In the following sections, you'll explore why this is the case, and some other stabilization techniques that are needed in order to get the above code to work. Below, you can find an implementation of the replay buffer, as well the training loop that we use to train DDPG.

# In[8]:


# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, state, next_state, action, reward, done):
        if len(self.storage) < self.max_size:
            self.storage.append((state, next_state, action, reward, done))
        else:
            # Remove random element in the memory beforea adding a new one
            self.storage.pop(random.randrange(len(self.storage)))
            self.storage.append((state, next_state, action, reward, done))


    def sample(self, batch_size=100, flat=True):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, next_states, actions, rewards, dones = [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done = self.storage[i]

            if flat:
                states.append(np.array(state, copy=False).flatten())
                next_states.append(np.array(next_state, copy=False).flatten())
            else:
                states.append(np.array(state, copy=False))
                next_states.append(np.array(next_state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            dones.append(np.array(done, copy=False))

        # state_sample, action_sample, next_state_sample, reward_sample, done_sample
        return {
            "state": np.stack(states),
            "next_state": np.stack(next_states),
            "action": np.stack(actions),
            "reward": np.stack(rewards).reshape(-1,1),
            "done": np.stack(dones).reshape(-1,1)
        }


# In[9]:


seed_ = 123
max_timesteps = 500 
batch_size = 64
discount = 0.99
eval_freq = 5e3
file_name = 'dt-class-rl'
start_timesteps = 1e4
expl_noise = 0.1


# In[10]:


import os

local_env = launch_env()
local_env = wrap_env(local_env)
local_env = ResizeWrapper(local_env)
local_env = ImgWrapper(local_env)

if not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

# Set seeds
seedall(seed_)

state_dim = local_env.observation_space.shape
action_dim = local_env.action_space.shape[0]
max_action = float(local_env.action_space.high[0])

# Initialize policy
policy = DDPGAgent(state_dim, action_dim, max_action)

replay_buffer = ReplayBuffer()

# Evaluate untrained policy
evaluations= [evaluate_policy(local_env, policy)]

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
episode_reward = None
env_counter = 0
while total_timesteps < max_timesteps:
    if done:
        if total_timesteps != 0:
            print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                total_timesteps, episode_num, episode_timesteps, episode_reward))
            policy.train(replay_buffer, episode_timesteps, batch_size, discount)

        # Evaluate episode
        if timesteps_since_eval >= eval_freq:
            timesteps_since_eval %= eval_freq
            evaluations.append(evaluate_policy(local_env, policy))

            policy.save(file_name, directory="./pytorch_models")
            np.savez("./pytorch_models/{}.npz".format(file_name),evaluations)

        # Reset environment
        env_counter += 1
        obs = local_env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # Select action randomly or according to policy
    if total_timesteps < start_timesteps:
        action = local_env.action_space.sample()
    else:
        action = policy.predict(np.array(obs))
        if expl_noise != 0:
            action = (action + np.random.normal(
                0,
                expl_noise,
                size=local_env.action_space.shape[0])
            ).clip(-1, +1)

    # Perform action
    new_obs, reward, done, _ = local_env.step(action)

    if episode_timesteps >= env_timesteps:
        done = True

    done_bool = 0 if episode_timesteps + 1 == env_timesteps else float(done)
    episode_reward += reward

    # Store data in replay buffer
    replay_buffer.add(obs, new_obs, action, reward, done_bool)

    obs = new_obs

    episode_timesteps += 1
    total_timesteps += 1
    timesteps_since_eval += 1

# Final evaluation
evaluations.append(evaluate_policy(local_env, policy))

if args.save_models:
    policy.save(file_name, directory="./pytorch_models")
np.savez("./pytorch_models/{}.npz".format(file_name),evaluations)


# # Stabilizing DDPG
# 
# As you may notice, the above model performs poorly or doesn't converge. Your job is to improve it; first in the notebook, later in the AIDO submission. This last part of the assignment consists of four sections:
# 
# **1. There are subtle, but important, bugs that have been introduced into the code above. Your job is to find them, and explain them in your `reinforcement-learning-answers.txt`. You'll want to reread the original [DQN](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning) and [DDPG](https://arxiv.org/abs/1509.02971) papers in order to better understand the issue, but by answering the following subquestions (*please put the answers to these in the submission for full credit*), you'll be on the right track:**
# 
#    a) Read some literature on actor-critic methods, including the original [actor-critic](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf) paper. What is an issue that you see related to *non-stationarity*? Define what _non-stationarity_ means in the context of machine learning and how it relates to actor-critic methods. In addition, give some hypotheses on why reinforcement learning is much more difficult (from an optimization perspective) than supervised learning, and how the answer to the previous question and this one are related.
# 
#    b) What role does the replay buffer play in off-policy reinforcement learning? It's most important parameter is `max_size` - how does changing this value (answer for both increasing and decreasing trends) qualitatively affect the training of the algorithm?
# 
#    c) **Challenge Question:** Briefly, explain how automatic differentiation works. In addition, expand on the difference between a single-element tensor (that `requires_grad`) and a scalar value as it relates to automatic differentiation; when do we want to backpropogate through a single-element tensor, and when do we not? Take a close look at the code and how losses are being backpropogated. On paper or your favorite drawing software, draw out the actor-critic architecture *as described in the code*, and label how the actor and critic losses are backpropogated. On your diagram, highlight the particular loss that will cause issues with the above code, and fix it.
#    
# For the next section, please pick **either** the theoretical or the practical pathway. If you don't have access to the necessary compute, for the exercise, please do the theoretical portion. 
#    
# _Theoretical Component_ 
# 
# **2. We discussed a case study of DQN in class. The original authors used quite a few tricks to get this to work. Detail some of the following, and explain what problem they solve in training the DQN:**
# 
# a) Target Networks
# 
# b) Annealed Learning Rates
# 
# c) Replay Buffer
# 
# d) Random Exploration Period
# 
# e) Preprocessing the Image
# 
# 
# **3. Read about either [TD3](https://arxiv.org/abs/1802.09477) or [Soft Actor Critic](https://arxiv.org/abs/1801.01290); for your choice, summarize what problems they are addressing with the standard actor-critic formulation, and how they solve them**
# 
# 
# _Practical Component_ 
# 
# **2. [Optional - if you have access to compute] Using your analysis from the reward engineering ablation, train two agents (after you've found the bugs in DDPG) - one with the standard, `gym-duckietown` reward, and another with the parameters of your choosing. Report each set of parameters, and describe qualitatively what type of behavior the agent produces.**
# 
# If you don't have the resources to actually train these agents, instead describe what types of behaviors each reward function might prioritize.
# 
# **3. [Optional - if you have access to compute] Using the instructions [here](http://docs.duckietown.org/DT19/AIDO/out/embodied_rl.html), use the saved policy files from this notebook and submit using the template submission provided through the AIDO submission. Report your best submission number (i.e the one you'd like to be graded) in `reinforcement-learning-answers.txt`**

# In[ ]:




