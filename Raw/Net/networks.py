import torch.nn as nn
import torch
import pysc2.lib.actions as actions
from pysc2.env import sc2_env
import torch.nn.functional as F
from torch.distributions import Categorical

class SpatialFeatures(nn.Module):
    def __init__(self, in_channels, hidden_channels = 16, n_channels = 32):
        super(SpatialFeatures, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, n_channels, kernel_size=3, padding='same'),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)

class NonSpatialFeatures(nn.Module):
    def __init__(self, n_features=256, n_channels=32, resolution=32):
        super(NonSpatialFeatures, self).__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * (resolution // 8) ** 2, n_features),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.net(x)
        return x
    
class NonSpatialFeatures_v1(nn.Module):
    def __init__(self, n_features=256, n_channels=32, resolution=32):
        super(NonSpatialFeatures_v1, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (resolution) ** 2, n_features),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.net(x)
        return x

class Actor(nn.Module):
    def __init__(self, n_features, action_space):
        super(Actor, self).__init__()
        self.linear = nn.Linear(n_features, action_space)
    
    def forward(self, x):
        return self.linear(x)

class Critic(nn.Module):
    def __init__(self, n_features):
        super(Critic, self).__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        return self.linear(x)

class CategoricalNet(nn.Module):
    def __init__(self, n_features, linear_size, hidden=256):
        super(CategoricalNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, linear_size),
        )
        
    def forward(self, state):
        x = self.net(state)
        log_probs = F.log_softmax(x, dim=-1)
        probs = torch.exp(log_probs)
        args = Categorical(probs).sample()
        return args.reshape(-1, 1), log_probs[range(len(args)), args], log_probs

class SpatialParameters(nn.Module):
    def __init__(self, n_channels, linear_size):
        super(SpatialParameters, self).__init__()
        self.size = linear_size
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding='same'),
        )
    def forward(self, state):
        batch_size = state.shape[0]
        x = self.net(state)
        x = x.reshape(batch_size, -1)
        log_probs = F.log_softmax(x, dim=-1)
        probs = torch.exp(log_probs)
        index = Categorical(probs).sample()
        x = index % self.size
#         index //= self.size
        y = index // self.size
        args = [[xi.item(), yi.item()] for xi, yi in zip(x, y)]
        return args, log_probs[range(batch_size), index], log_probs
    
class ActorCritic(nn.Module):
    def __init__(self, env, spatial_model, nonspatial_model, spatial_dict, nonspatial_dict, 
                 n_features, n_channels, action_dict=None):
        super(ActorCritic, self).__init__()
        self.action_dict = action_dict
        self.resolution = env.observation_spec()[0]['feature_screen'][1:]
        self.all_actions = env.action_spec()[0][1]
        
        self.n_features = n_features
        self.n_channels = n_channels
        
        self.spatial_net = spatial_model(**spatial_dict)
        self.nonspatial_net = nonspatial_model(**nonspatial_dict) 
        
        self.actor_net = Actor(self.n_features, len(action_dict))
        self.critic_net = Critic(self.n_features)
        self._init_params_nets()
        
    def _init_params_nets(self):
        self.act_args = {} 
        self.act_ids = {}
        self.args_dict = {}
        self.args_type = {}
        arguments_nets = {}

        for i, a in enumerate(self.action_dict.keys()):
            action = self.all_actions[self.action_dict[a]]
            args = action.args
            self.act_args[a] = [arg for arg in args]
            self.act_ids[i] = action.id
            for arg in args:
                self.args_dict[arg.name] = arg.id
                
                sizes = arg.sizes
                if (len(sizes) == 1):
                    arguments_nets[arg.name] = CategoricalNet(self.n_features, sizes[0])
                    self.args_type[arg.name] = 'categorical'
                else:
                    arguments_nets[arg.name] = SpatialParameters(self.n_channels, sizes[0])
                    self.args_type[arg.name] = 'spatial'
        
        self.arguments_nets = nn.ModuleDict(arguments_nets)
    
    def actor(self, state, action_mask):
        spatial_features = self.spatial_net(state)
        nonspatial_features = self.nonspatial_net(spatial_features)
        x = self.actor_net(nonspatial_features)
        log_probs = F.log_softmax(x.masked_fill((action_mask).bool(), float('-inf')), dim=-1)
        return log_probs, spatial_features, nonspatial_features
    
    def critic(self, state):
        spatial_features = self.spatial_net(state)
        nonspatial_features = self.nonspatial_net(spatial_features)
        V = self.critic_net(nonspatial_features)
        return V
    
    def get_arguments(self, state, arg_name):
        return self.arguments_nets[arg_name](state)

class ActorCritic_v2(ActorCritic):
    def __init__(self, env, spatial_model, nonspatial_model, spatial_dict, nonspatial_dict, 
                 n_features, n_channels, action_dict=None):
        super(ActorCritic_v2, self).__init__(env, spatial_model, nonspatial_model, spatial_dict, nonspatial_dict, 
                 n_features, n_channels, action_dict)
        
    def _init_params_nets(self):
        self.act_args = {} 
        self.act_ids = {}
        self.args_dict = {}
        self.args_type = {}
        arguments_nets = {}

        for i, a in enumerate(self.action_dict.keys()):
            action = self.all_actions[self.action_dict[a]]
            args = action.args
            self.act_args[a] = [arg for arg in args]
            self.act_ids[i] = action.id
            for arg in args:
                name = action.name + arg.name
                
                self.args_dict[arg.name] = arg.id
                
                sizes = arg.sizes
                if (len(sizes) == 1):
                    arguments_nets[name] = CategoricalNet(self.n_features, sizes[0])
                    self.args_type[name] = 'categorical'
                else:
                    arguments_nets[name] = SpatialParameters(self.n_channels, sizes[0])
                    self.args_type[name] = 'spatial'
        
        self.arguments_nets = nn.ModuleDict(arguments_nets) 