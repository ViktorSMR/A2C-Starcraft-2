from pysc2.env import sc2_env
from pysc2.lib import actions 
import numpy as np
from Net.actorCritic import A2C
from Obs.obs import *
from tqdm import trange
import torch
from IPython.display import clear_output
import matplotlib.pyplot as plt
import os

def get_unique_ID(map_name):
    if not os.path.exists("Results\\" + map_name):
        os.makedirs("Results\\" + map_name)
    ID = len([name for name in os.listdir("Results\\" + map_name) if os.path.isdir("Results\\" + map_name + "\\" + name)])
    ID = str(ID)
    return ID

def get_action_dict(action_names):
    action_ids = [actions.FUNCTIONS[a_name].id for a_name in action_names]
    action_dict = {i:action_ids[i] for i in range(len(action_ids))}
    return action_dict

def get_action_mask(available_actions, action_dict):
    action_mask = ~np.array([action_dict[i] in available_actions for i in action_dict.keys()])
    return action_mask

def init_game(game_params, map_name="MoveToBeacon", step_multiplier=8, **kwargs):
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(["run"])
    race = sc2_env.Race(1)
    agent = sc2_env.Agent(race, "Agent")
    agent_interface_format = sc2_env.parse_agent_interface_format(**game_params)
    
    game_params = dict(map_name=map_name, 
                       players=[agent],
                       game_steps_per_episode = 0,
                       step_mul = step_multiplier,
                       agent_interface_format=[agent_interface_format]
                       )  
    env = sc2_env.SC2Env(**game_params, **kwargs)
    return env

class Env():
    def __init__(self, game_params, map_name, obs_proc_params, action_dict, **kwargs):
        self.env = init_game(game_params, map_name, **kwargs)
        self.op = ObsProcesser(**obs_proc_params)
        self.action_dict = action_dict
    def step(self, actions):
        obs = self.env.step(actions)
        new_state_dict = self.op.get_state(obs)
        new_state = merge_screen_and_minimap(new_state_dict)
        reward = obs[0].reward
        done = obs[0].last()
        
        if done:
            bootstrap = True
        else:
            bootstrap = False
            
        if done:
            obs = self.env.reset()
            state_dict = self.op.get_state(obs)
            state = merge_screen_and_minimap(state_dict)
        else:
            state = new_state
        
        available_actions = obs[0].observation.available_actions
        action_mask = get_action_mask(available_actions, self.action_dict)
        results = [(state, reward, done, bootstrap, action_mask)]
        states, rews, dones, bootstraps, action_mask = zip(*results)
        return np.stack(states), np.stack(rews), np.stack(dones), np.stack(bootstraps), np.stack(action_mask)
    def reset(self):
        obs = self.env.reset()
        state_dict = self.op.get_state(obs)
        state = merge_screen_and_minimap(state_dict)
        available_actions = obs[0].observation.available_actions
        action_mask = get_action_mask(available_actions, self.action_dict)
        results = [(state, action_mask)]
        states, action_mask = zip(*results)
        return np.stack(states), np.stack(action_mask)
    def close(self):
        self.env.close()

def evaluate(env, agent, n_games=5):
    rewards = []
    for i in range(n_games):
        state, action_mask = env.reset()
        sum_reward = 0
        while True:
            action, _, _ = agent.step(state, action_mask)
            new_state, reward, done, bootstrap, action_mask = env.step(action)
            sum_reward += reward
            state = new_state
            if done:
                break
        rewards.append(sum_reward)
    return np.mean(rewards)

def train(agent, test_env, env, map_name, n_steps, max_n_steps, steps_count, loss_freq, step=0, mean_rw_history=[], loss_history = []):
    ID = get_unique_ID(map_name)
    print("process ID: ", ID),     
    state, action_mask = env.reset()
    optimizer = torch.optim.RMSprop(agent.AC.parameters(), lr=7e-4)
    for step in trange(step, max_n_steps):
        
        states = state
        
        action, log_prob, entropy = agent.step(state, action_mask)
        new_state, reward, done, bootstrap, action_mask = env.step(action)
        
#         g_t = reward
#         s_log_prob = log_prob
#         s_entropy = entropy
        
        rewards = reward
        done = [False]
        log_probs = log_prob
        entropys = entropy.reshape(-1)
        
        state = new_state
        
        for t in range(1, n_steps):
            states = np.append(states, state, axis=0)
            action, log_prob, entropy = agent.step(state, action_mask)
            new_state, reward, done, bootstrap, action_mask = env.step(action)
            rewards = np.append(rewards, reward, axis=0)
            log_probs = torch.cat((log_probs, log_prob), 0)
            entropys = torch.cat((entropys, entropy.reshape(-1)), 0)
            
            state = new_state
            
            last_state = state
            
            if (done):
#                 g_t += agent.AC.critic(torch.from_numpy(state).float().to(agent.device)).detach().cpu().numpy()[0] * agent.gamma**(t + 1)
                break

        
        actor_loss, critic_loss = agent.compute_losses(states, last_state, rewards, done, log_probs, entropys)

        loss = actor_loss + critic_loss * agent.beta
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step % loss_freq == 0):
            loss_history.append(loss.data.cpu().item())
        
        if (step % steps_count == 0):
            mean_rw_history.append(evaluate(test_env, agent))
            
            clear_output(True)
            plt.figure(figsize=[16, 9])
            
            plt.subplot(2, 2, 1)
            plt.title("Mean reward per episode")
            plt.plot(mean_rw_history)
            plt.grid()
            
            plt.subplot(2, 2, 2)
            plt.title("loss history")
            plt.plot(loss_history)
            plt.grid()
            
            plt.show()
            
            if not os.path.exists("Results\\" + map_name + "\\" + ID):
                os.makedirs("Results\\" + map_name + "\\" + ID)
            torch.save(mean_rw_history, "Results\\" + map_name + "\\" + ID + "\\mean_rw_history.pth")
            
            if not os.path.exists("Results\\" + map_name + "\\" + ID):
                os.makedirs("Results\\" + map_name + "\\" + ID)
            torch.save(loss_history, "Results\\" + map_name + "\\" + ID + "\\loss_history.pth")
            
            if not os.path.exists("Results\\" + map_name + "\\" + ID):
                os.makedirs("Results\\" + map_name + "\\" + ID)
            torch.save(agent, "Results\\" + map_name + "\\" + ID + "\\model.pth")