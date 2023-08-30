from pysc2.env import sc2_env
from pysc2.lib import actions 
import numpy as np

def merge_screen_and_minimap(state_dict):
    screen = state_dict['screen_layers']
    minimap = state_dict['minimap_layers']
    if (len(minimap) > 0):
        if (len(screen) > 0):
            state = np.concatenate([screen, minimap])
        else:
            state = minimap
    else:
        state = screen
    return state

class ObsProcesser():
    def __init__(self, screen_names=[], minimap_names=[], select_all=True):
        self.screen_var = { 'visibility_map': (1,'ohe', 2),
                        'player_relative': (5, 'ohe', 3), #friendly (1), neutral(3), enemy(4)
                        'unit_type': (6,'ohe', 13),
                        'selected': (7,'ohe', 1),
                        'unit_hit_points': (8,'log'), 
                        'unit_hit_points_ratio': (9,'log'), 
                        'unit_density': (14, 'float'), 
                        'unit_density_aa': (15, 'float'),
                        'pathable': (24, 'ohe', 1),  
                        'buildable': (25, 'ohe', 1)  }
        self.minimap_var = { 'visibility_map': (1, 'ohe', 2),
                         'camera': (3, 'ohe', 1), 
                         'player_relative': (5, 'ohe', 3),
                         'selected': (6,'ohe', 1),  
                         'pathable': (9, 'ohe', 1),  
                         'buildable': (10,'ohe', 1) }
        self.possible_values = {
             'visibility_map': np.array([1,2]),
             'player_relative': np.array([1,3,4]),
             'unit_type':np.array([9,18,19,20,21,45,48,105,110,317,341,342,1680]),
             'selected': np.array([1]),
             'pathable': np.array([1]),
             'buildable': np.array([1]),
             'camera': np.array([1]) }
        if (select_all):
            self.screen_names = [k for k in self.screen_var.keys()]
            self.minimap_names = [k for k in self.minimap_var.keys()]
        else:
            self.screen_names = screen_names 
            self.minimap_names = minimap_names
        
        self.screen_indexes = [self.screen_var[i][0] for i in self.screen_names]
        self.minimap_indexes = [self.minimap_var[i][0] for i in self.minimap_names]
        
    def get_state(self, obs):
        feature_screen = obs[0].observation['feature_screen']
        feature_minimap = obs[0].observation['feature_minimap']
        
        screen_layers = self._process_screen_features(feature_screen)
        minimap_layers = self._process_minimap_features(feature_minimap)
        
        state = {'minimap_layers': minimap_layers, 'screen_layers': screen_layers}
        return state
        
    def get_n_channels(self):
        screen_channels, minimap_channels = 0, 0
        
        for name in self.screen_names:
            if self.screen_var[name][1] == 'ohe':
                screen_channels +=  self.screen_var[name][2]
            elif (self.screen_var[name][1] == 'log') or (self.screen_var[name][1]=='float'):
                screen_channels += 1
                
        for name in self.minimap_names:
            if self.minimap_var[name][1] == 'ohe':
                minimap_channels +=  self.minimap_var[name][2]
            elif (self.minimap_var[name][1] == 'log') or (self.minimap_var[name][1]=='float'):
                minimap_channels += 1
                
        return screen_channels, minimap_channels
    
    def _process_screen_features(self, features):
        names = list(features._index_names[0].keys())
        processed_layers = []
        
        for i, idx in enumerate(self.screen_indexes):
            layer = np.array(features[idx])
            
            if (self.screen_var[names[idx]][1] == 'ohe'):
                layer = self._process_ohe_layer(layer, names[idx])
            elif (self.screen_var[names[idx]][1] == 'float'):
                layer = self._process_float_layer(layer)
            elif (self.screen_var[names[idx]][1] == 'log'):
                layer = self._process_log_layer(layer)
        
            processed_layers.append(layer)
            
        if len(processed_layers) > 0:
            processed_layers = np.concatenate(processed_layers).astype(float)
        
        return processed_layers
    
    def _process_minimap_features(self, features):
        names = list(features._index_names[0].keys())
        processed_layers = []
        
        for i, idx in enumerate(self.minimap_indexes):
            layer = np.array(features[idx])
            
            if (self.minimap_var[names[idx]][1] == 'ohe'):
                layer = self._process_ohe_layer(layer, names[idx])
            elif (self.minimap_var[names[idx]][1] == 'float'):
                layer = self._process_float_layer(layer)
            elif (self.minimap_var[names[idx]][1] == 'log'):
                layer = self._process_log_layer(layer)
        
            processed_layers.append(layer)
            
        if len(processed_layers) > 0:
            processed_layers = np.concatenate(processed_layers).astype(float)
        
        return processed_layers
    
    def _process_float_layer(self, layer):
        return layer.reshape((1, ) + layer.shape[-2:]).astype(float)
    
    def _process_log_layer(self, layer):
        mask = layer != 0
        layer[mask] = np.log2(layer[mask])
        return layer.reshape((1, ) + layer.shape[-2:]).astype(float)
    
    def _process_ohe_layer(self, layer, name):
        possible_values = self.possible_values[name]
        ohe_layer = (layer == possible_values.reshape(-1, 1, 1)).astype(float)
        return ohe_layer