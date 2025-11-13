import numpy as np
import gym
from gym import spaces
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad_vec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class DynamicSystemEnv(gym.Env):
    def __init__(self):
        # parameters
        self.T = 0.05
        self.kus = 232.5e3
        self.mus = 65
        self.ms = 325
        self.ks = 27692.0
        self.cs = 1906.5
        self.ct = 0
        self.ks_max = 40000
        self.ks_min = 10000
        self.cs_max = 3000
        self.cs_min = 500

        A, B, Gd = self.sys_matrices()
        self.A = A
        self.B = B
        self.Gd = Gd

        # System dynamics parameters
        self.Q = np.array([[10,0,0,0], [0,1,0,0], [0,0,50,0], [0,0,0,5]])     # State cost matrix
        self.R = 0.000001                                 # Action cost
        
        # State and action spaces
        self.observation_space = spaces.Box(low=-2.5, high=2.5, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1000, high=1000, shape=(1,), dtype=np.float32)
        
        # Initial state
        self.state = np.array([-0.1,-1,-0.1,-0.5])
        
        # Maximum episode steps
        self.max_steps = 100
        self.current_step = 0
    
    def sys_matrices(self):
        Ac = np.array([[0,1,0,0], [-self.kus/self.mus,-self.cs/self.mus,self.ks/self.mus,self.cs/self.mus],[0,-1,0,1],[0,self.cs/self.ms,-self.ks/self.ms,-self.cs/self.ms]])
        Bc = np.array([[0],[-1/self.mus],[0],[1/self.ms]])
        Gc = np.array([[-1],[self.ct/self.mus],[0],[0]])
        def funB(t):
            return expm(Ac * t) @ Bc

        def funG(t):
            return expm(Ac * t) @ Gc

        A = expm(Ac * self.T)
        B = quad_vec(funB, 0, self.T)
        B = B[0]
        Gd = quad_vec(funG, 0, self.T)
        Gd = Gd[0]
        return A, B, Gd

    def step(self, action, ks_normalized, cs_normalized):
        ks = ks_normalized*(self.ks_max-self.ks_min)+self.ks_min
        cs = cs_normalized*(self.cs_max-self.cs_min)+self.cs_min
        self.ks = ks
        self.cs = cs

        A, B, Gd = self.sys_matrices()
        self.A = A
        self.B = B
        self.Gd = Gd

        # Clip action to stay within action bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Update state based on system dynamics
        x = self.state
        # x_next = np.dot(self.A, x) + np.dot(self.B, action) is the deterministic part
        x_next = np.dot(self.A, x) + np.dot(self.B, action) + np.dot(self.Gd, np.random.normal(loc=0, scale=0.3, size=(1, )))
        
        
        # Calculate reward      
        reward = -x_next@self.Q@x_next.T - self.R * action**2
                
        # Update state
        self.state = x_next
        
        # Check if done
        done = (self.current_step >= self.max_steps) or (self.state[0]**2+self.state[1]**2+self.state[2]**2+self.state[3]**2<=0.01)
        self.current_step += 1
        
        return self.state, reward[0], done, {}
    
    def step_real(self, action, dzdt, ks_normalized, cs_normalized):
        ks = ks_normalized*(self.ks_max-self.ks_min)+self.ks_min
        cs = cs_normalized*(self.cs_max-self.cs_min)+self.cs_min
        self.ks = ks
        self.cs = cs
        knl = 0.01*ks # nonlinear term of spring coefficient
        cnl = 0.01*cs # nonlinear term of damping coefficient

        A, B, Gd = self.sys_matrices()
        self.A = A
        self.B = B
        self.Gd = Gd

        # Clip action to stay within action bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Update state based on system dynamics
        x = self.state
        x_next = np.dot(self.A, x) + np.dot(self.B, action) + np.dot(self.Gd, dzdt) \
            + np.array([0,-1/self.mus*(knl*x[2]**3+cnl*abs(x[3]-x[1])*(x[3]-x[1])),0,-1/self.ms*(knl*x[2]**3+cnl*abs(x[3]-x[1])*(x[3]-x[1]))])

        # Calculate reward
        reward = -x_next@self.Q@x_next.T - self.R * action**2
                
        # Update state
        self.state = x_next
        
        # Check if done
        done = (self.current_step >= self.max_steps) or (self.state[0]**2+self.state[1]**2+self.state[2]**2+self.state[3]**2<=0.01)
        self.current_step += 1
        
        return self.state, reward[0], done, {}
    
    def reset(self,ks_normalized,cs_normalized):
        # Reset state to initial value
        ks = ks_normalized*(self.ks_max-self.ks_min)+self.ks_min
        cs = cs_normalized*(self.cs_max-self.cs_min)+self.cs_min
        self.ks = ks
        self.cs = cs
        A, B, Gd = self.sys_matrices()
        self.A = A
        self.B = B
        self.Gd = Gd
        self.state = np.array([-0.1,-1,-0.1,-0.5])
        self.current_step = 0
        return self.state
    
    def reset_origin(self,ks_normalized,cs_normalized):
        # Reset state to the origin
        ks = ks_normalized*(self.ks_max-self.ks_min)+self.ks_min
        cs = cs_normalized*(self.cs_max-self.cs_min)+self.cs_min
        self.ks = ks
        self.cs = cs
        A, B, Gd = self.sys_matrices()
        self.A = A
        self.B = B
        self.Gd = Gd
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.current_step = 0
        return self.state
    
    def render(self, mode='human'):
        pass

    def close(self):
        pass
