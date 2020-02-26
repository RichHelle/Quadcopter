import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.runtime = runtime


        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # Original Reward Function
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        # More heavily punishes variation in x and y directions 
        
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state

    

class Takeoff(Task):
    """Takeoff that defines the goal as starting at [0,0,0] and reaching [0,0, target_height] while prodiving feedback to the agent."""
    """Initialize a Takeoff object.
    Params
    ======
        init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
        init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
        init_angle_velocities: initial radians/second for each of the three Euler angles
        runtime: time limit for each episode
        target_pos: target/goal (x,y,z) position for the agent
    """
    def __init__(self, init_pose=None, init_velocities=None, 
                   init_angle_velocities=None, runtime=5., target_pos=None):

        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward_takeoff() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def get_reward_takeoff(self):
        """Uses current pose of sim to return reward. Takeoff specific reward.
           Reward function also discourages 
           """
        #Original Reward function
        #reward = 1. - .3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        
        z_c  = 0.2
        xy_c = 0.1
        e_c  = 0.05
        
        
        #xy_c - Applies penalty for straying from origin.
        #z_c  - Applies penalty for being away from target height
        #e_c - Applies penalty for non-zero euler angles (note, starting pose is 0 for all three so no inherent penalty.)
        
        reward = 5.0 \
                 - xy_c * (abs(self.sim.pose[:2]-self.target_pos[:2])).sum() \
                 - z_c  * (abs(self.sim.pose[2]-self.target_pos[2])).sum()   \
                 - e_c  * (abs(self.sim.pose[3:])).sum()

        
        # Possible alternative rewards
        
        # More heavily punishes variation in x and y to encourage a straight takeoff
        # reward = 1. - .5*(abs(self.sim.pose[:2] - self.target_pos[:2])).sum() - .3*(abs(self.sim.pose[2] - self.target_pos[2])).sum() 
        
        # Variation in x and y consistently, allows z component penalty is time dependent. 
        # reward = 1. - .3*(abs(self.sim.pose[:2] - self.target_pos[:2])).sum() - .3*(abs(self.sim.pose[2] - 
                       #(self.target_pos[2] * self.sim.time / (self.sim.runtime - .1 )).sum() 
                
        return reward

     

    