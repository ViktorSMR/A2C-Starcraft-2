from Net.networks import *
import numpy as np

class A2C():
    def __init__(self, env, spatial_model, nonspatial_model, spatial_dict, 
                 nonspatial_dict, n_features, n_channels, gamma=0.99, beta=0.1, action_dict=None, eta=0.1, device='cuda'): 
        self.eta = eta
        self.gamma = gamma
        self.beta = beta
        self.AC = ActorCritic(env, spatial_model, nonspatial_model, spatial_dict,
                              nonspatial_dict, n_features, n_channels, action_dict)
        self.device = device
        self.AC.to(self.device)
    
    def step(self, state, action_mask):
        state = torch.from_numpy(state).float().to(self.device)
        action_mask = torch.tensor(action_mask).to(self.device)
        
        log_probs, spatial_features, nonspatial_features = self.AC.actor(state, action_mask)
        entropy = self.compute_entropy(log_probs)
        probs = torch.exp(log_probs)
        acts = Categorical(probs).sample()
        log_probs = log_probs[range(len(acts)), acts]
        
        args, args_log_probs, args_entropy = self.get_arguments(spatial_features, nonspatial_features, acts)
        log_probs = log_probs + args_log_probs
        entropy = entropy + args_entropy

        action_id = np.array([self.AC.act_ids[act.item()] for act in acts])
        action = [actions.FunctionCall(action_id[i], args[i]) for i in range(len(action_id))]

        return action, log_probs, entropy
    
    def get_arguments(self, spatial_features, nonspatial_features, actions):
        action_args, args_log_prob, args_entropy = [], [], []
        for action in actions:
            action = action.item()
            args = self.AC.act_args[action]
            
            action_arg = []
            sarg_log_prob = torch.tensor([0]).float().to(self.device)
            sarg_arg_entropy = torch.tensor([0]).float().to(self.device)
            for arg in args:
                arg = arg.name
                if (self.AC.args_type[arg] == 'spatial'):
                    act_arg, arg_log_prob, arg_log_probs = self.AC.get_arguments(spatial_features, arg)
                    entropy = self.compute_entropy(arg_log_probs)
                elif (self.AC.args_type[arg] == 'categorical'):
                    act_arg, arg_log_prob, arg_log_probs = self.AC.get_arguments(nonspatial_features, arg)
                    entropy = self.compute_entropy(arg_log_probs)
                action_arg.append(torch.tensor(act_arg).reshape(-1).detach().cpu().numpy())
                sarg_log_prob += arg_log_prob
                sarg_arg_entropy += entropy
            
            args_log_prob.append(sarg_log_prob)
            args_entropy.append(sarg_arg_entropy)
            action_args.append(action_arg)
        
        args_log_prob = torch.stack(args_log_prob, axis=0).squeeze()
        args_entropy = torch.stack(args_entropy, axis=0).squeeze()
        return action_args, args_log_prob, args_entropy
        
    def compute_entropy(self, log_probs):
        log_probs_copy = torch.masked_fill(log_probs, (log_probs==float('-inf')), 0)
        probs = torch.exp(log_probs_copy)
        mult = probs * log_probs_copy
        if (len(probs.shape) == 2):
            entropy = mult.sum(axis=1)
        else:
            entropy = mult.sum(axis=(1, 2))
        return entropy
    
    def compute_losses(self, states, last_state, rewards, done, log_probs, entropys):
        states = torch.from_numpy(states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        last_state = torch.from_numpy(last_state).float().to(self.device)
        
        if done[0]:
            R = torch.zeros(1, 1).to(self.device)
        else:
            R = self.AC.critic(last_state)
        
        actor_loss = torch.zeros(1, 1).to(self.device)
        critic_loss = torch.zeros(1, 1).to(self.device)
        
        for state, rw, log_prob, entropy in zip(reversed(states), reversed(rewards), reversed(log_probs), reversed(entropys)):
            state = state.unsqueeze(0)
            rw = rw.unsqueeze(0)
            log_prob = log_prob.unsqueeze(0)
            entropy = entropy.unsqueeze(0)
            R = rw + self.gamma * R
            
            advantage = R - self.AC.critic(state)
            
            actor_loss = actor_loss - log_prob * (advantage.detach()) - self.eta * entropy
            critic_loss = critic_loss + (advantage)**2
        
        return actor_loss, critic_loss
    
class A2C_v2(A2C):
    def __init__(self, env, spatial_model, nonspatial_model, spatial_dict, 
                 nonspatial_dict, n_features, n_channels, gamma=0.99, beta=0.1, action_dict=None, eta=0.1, device='cuda'): 
        self.eta = eta
        self.gamma = gamma
        self.beta = beta
        self.AC = ActorCritic_v2(env, spatial_model, nonspatial_model, spatial_dict,
                              nonspatial_dict, n_features, n_channels, action_dict)
        self.device = device
        self.AC.to(self.device)
        
    def get_arguments(self, spatial_features, nonspatial_features, actions):
        action_args, args_log_prob, args_entropy = [], [], []
        for action in actions:
            action = action.item()
            args = self.AC.act_args[action]
            
            action_arg = []
            sarg_log_prob = torch.tensor([0]).float().to(self.device)
            sarg_arg_entropy = torch.tensor([0]).float().to(self.device)
            for arg in args:
                arg = self.AC.all_actions[self.AC.action_dict[action]].name + arg.name
                if (self.AC.args_type[arg] == 'spatial'):
                    act_arg, arg_log_prob, arg_log_probs = self.AC.get_arguments(spatial_features, arg)
                    entropy = self.compute_entropy(arg_log_probs)
                elif (self.AC.args_type[arg] == 'categorical'):
                    act_arg, arg_log_prob, arg_log_probs = self.AC.get_arguments(nonspatial_features, arg)
                    entropy = self.compute_entropy(arg_log_probs)
                action_arg.append(torch.tensor(act_arg).reshape(-1).detach().cpu().numpy())
                sarg_log_prob += arg_log_prob
                sarg_arg_entropy += entropy
            
            args_log_prob.append(sarg_log_prob)
            args_entropy.append(sarg_arg_entropy)
            action_args.append(action_arg)
        
        args_log_prob = torch.stack(args_log_prob, axis=0).squeeze()
        args_entropy = torch.stack(args_entropy, axis=0).squeeze()
        return action_args, args_log_prob, args_entropy
        