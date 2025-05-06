# Used code from
# DQN implementation by Tejas Kulkarni found at
# https://github.com/mrkulk/deepQN_tensorflow

# Used code from:
# The Pacman AI projects were developed at UC Berkeley found at
# http://ai.berkeley.edu/project_overview.html


import numpy as np
import random
import util
from util import manhattanDistance  # Add import for manhattanDistance
import time
import sys
import os
import csv

# Pacman game
from pacman import Directions
from game import Agent
import game

# Replay memory
from collections import deque

# Neural nets
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from DQN import *

params = {
    # Model backups
    'load_file': None,
    'save_file': 'default',   
    
    # Training parameters
    'train_start': 3000,    
    'batch_size': 64,       # Increased from 32 for more stable gradients (layers?)
    'mem_size': 100000,     # Replay memory size

    'discount': 0.95,       # Increased from 0.90 to value future rewards more
    'lr': .0002,            # Restored from .0001 to .0002 for faster learning

    # Epsilon value (epsilon-greedy)
    'eps': 1.0,             # Epsilon start value
    'eps_final': 0.05,      # Reduced from 0.1 to encourage more exploitation at end
    'eps_step': 7500,       # Increased from 10000 to decay epsilon more slowly
    
    # Debug parameters
    'debug': False,         
}                     



class PacmanDQN(game.Agent):
    def __init__(self, args):

        print("Initialise DQN Agent")

        # Load parameters from user-given arguments
        self.params = params
        self.params['width'] = args['width']
        self.params['height'] = args['height']
        self.params['num_training'] = args['numTraining']
        
        # Get transition mode directly from args if available
        if 'transition_mode' in args:
            self.transition_mode = args['transition_mode']
            self.params['save_file'] = self.transition_mode
            print(f"Setting save_file from args: {self.transition_mode}")
        else:
            self.transition_mode = 'default'
        
        # Create saves directory if it doesn't exist
        if not os.path.exists('saves'):
            os.makedirs('saves')
        
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # Setup CSV logging
        self.log_file_path = './logs/'+time.strftime("%Y%m%d-%H%M%S")+'-training-metrics.csv'
        self.csv_header = ['episode', 'total_reward', 'avg_q_value', 'min_q_value', 'max_q_value', 'win', 'steps', 'epsilon']
        with open(self.log_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_header)

        # Start Tensorflow session
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))
        self.qnet = DQN(self.params)

        # time started
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        # Q and cost
        self.Q_global = []
        self.cost_disp = 0     

        # Stats
        self.cnt = self.qnet.sess.run(self.qnet.global_step)
        self.local_cnt = 0

        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.

        self.replay_mem = deque()
        self.last_scores = deque()
        
        # Metrics for CSV logging
        self.ep_steps = 0
        self.episode_q_min = float('inf')
        self.episode_q_max = float('-inf')
        self.episode_q_sum = 0
        self.episode_q_count = 0


    def getMove(self, state):
        # Exploit / Explore
        if np.random.rand() > self.params['eps']:
            # Exploit action
            self.Q_pred = self.qnet.sess.run(
                self.qnet.y,
                feed_dict = {self.qnet.x: np.reshape(self.current_state,
                                                     (1, self.params['width'], self.params['height'], 6)), 
                             self.qnet.q_t: np.zeros(1),
                             self.qnet.actions: np.zeros((1, 4)),
                             self.qnet.terminals: np.zeros(1),
                             self.qnet.rewards: np.zeros(1),
                             self.qnet.keep_prob: 1.0})[0]

            # Track Q-values for logging
            q_max = max(self.Q_pred)
            q_min = min(self.Q_pred)
            self.Q_global.append(q_max)
            self.episode_q_min = min(self.episode_q_min, q_min)
            self.episode_q_max = max(self.episode_q_max, q_max)
            self.episode_q_sum += q_max
            self.episode_q_count += 1
            
            a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))

            if len(a_winner) > 1:
                move = self.get_direction(
                    a_winner[np.random.randint(0, len(a_winner))][0])
            else:
                move = self.get_direction(
                    a_winner[0][0])
        else:
            # Random:
            move = self.get_direction(np.random.randint(0, 4))

        # Save last_action
        self.last_action = self.get_value(move)

        return move

    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0.
        elif direction == Directions.EAST:
            return 1.
        elif direction == Directions.SOUTH:
            return 2.
        else:
            return 3.

    def get_direction(self, value):
        if value == 0.:
            return Directions.NORTH
        elif value == 1.:
            return Directions.EAST
        elif value == 2.:
            return Directions.SOUTH
        else:
            return Directions.WEST
            
    def observation_step(self, state):
        if self.last_action is not None:
            # current state
            self.last_state = np.copy(self.current_state)
            self.current_state = self.getStateMatrices(state)

            # current score
            self.current_score = state.getScore()
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            # REWARD SCALING - Enhanced reward shaping
            if reward > 20:
                self.last_reward = 50.0    # eat ghost - increased from 20
            elif reward > 0:
                self.last_reward = 10.0    # eat food - increased from 5
            elif reward < -10:
                self.last_reward = -200.0  # get eaten - more severe penalty than -100
                self.won = False
            elif reward < 0:
                self.last_reward = -1.0    # punish time - increased from -0.5
            
            # Add living penalty to encourage faster completion
            if reward == 0:
                self.last_reward = -0.1
                
            # Add proximity reward to encourage approaching food
            food_positions = state.getFood().asList()
            pacman_position = state.getPacmanPosition()
            if food_positions:
                min_food_distance = min([manhattanDistance(pacman_position, food) for food in food_positions])
                # Small bonus for getting closer to food
                if hasattr(self, 'last_min_food_distance') and min_food_distance < self.last_min_food_distance:
                    self.last_reward += 0.5
                self.last_min_food_distance = min_food_distance
            
            # Add ghost avoidance reward
            ghost_positions = state.getGhostPositions()
            if ghost_positions:
                min_ghost_distance = min([manhattanDistance(pacman_position, ghost) for ghost in ghost_positions])
                if min_ghost_distance < 3:  # Ghost is dangerously close
                    # Scale penalty inversely with distance
                    self.last_reward -= (3 - min_ghost_distance) * 2
            
            # Terminal state rewards
            if(self.terminal and self.won):
                self.last_reward = 200.0  # Increased win reward from 100
            self.ep_rew += self.last_reward

            # store last experience into memory
            experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()

            # Train
            self.train()

        # Next
        self.local_cnt += 1
        self.frame += 1
        self.ep_steps += 1
        self.params['eps'] = max(self.params['eps_final'],
                                 1.00 - float(self.cnt)/ float(self.params['eps_step']))


    def observationFunction(self, state):
        # Do observation
        self.terminal = False
        self.observation_step(state)
        
        # Print progress update every 500 episodes
        if self.numeps % 500 == 0 and self.frame == 0:
            print(f"Training progress: Episode {self.numeps}/{self.params['num_training']} ({self.numeps/self.params['num_training']*100:.1f}%)")

        return state

    def final(self, state):
        # Next
        self.ep_rew += self.last_reward

        # Do observation
        self.terminal = True
        self.observation_step(state)

        # Log metrics to CSV
        avg_q = self.episode_q_sum / max(1, self.episode_q_count)
        min_q = self.episode_q_min if self.episode_q_min != float('inf') else 0
        max_q = self.episode_q_max if self.episode_q_max != float('-inf') else 0
        
        with open(self.log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.numeps,          # episode
                self.ep_rew,          # total_reward
                avg_q,                # avg_q_value
                min_q,                # min_q_value
                max_q,                # max_q_value
                int(self.won),        # win
                self.ep_steps,        # steps
                self.params['eps']    # epsilon
            ])

        # Check if this is the last episode of training
        is_last_episode = self.numeps >= self.params['num_training']
        
        # Only save on the last episode (best version of model)
        if self.params['save_file'] and is_last_episode:
            # Save a clean final model with consistent naming
            final_path = f"saves/model-{self.params['save_file']}_final"
            self.qnet.save_ckpt(final_path)
            print(f"Final model saved: {final_path}")
        
        # Print stats
        log_file = open('./logs/'+str(self.general_record_time)+'-l-'+str(self.params['width'])+'-m-'+str(self.params['height'])+'-x-'+str(self.params['num_training'])+'.log','a')
        log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        log_file.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        sys.stdout.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.flush()
        
        # Reset episode metrics
        self.episode_q_min = float('inf')
        self.episode_q_max = float('-inf')
        self.episode_q_sum = 0
        self.episode_q_count = 0
        self.ep_steps = 0

    def train(self):
        # Train
        if (self.local_cnt > self.params['train_start']):
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_s = [] # States (s)
            batch_r = [] # Rewards (r)
            batch_a = [] # Actions (a)
            batch_n = [] # Next states (s')
            batch_t = [] # Terminal state (t)

            for i in batch:
                batch_s.append(i[0])
                batch_r.append(i[1])
                batch_a.append(i[2])
                batch_n.append(i[3])
                batch_t.append(i[4])
            batch_s = np.array(batch_s)
            batch_r = np.array(batch_r)
            batch_a = self.get_onehot(np.array(batch_a))
            batch_n = np.array(batch_n)
            batch_t = np.array(batch_t)

            self.cnt, self.cost_disp = self.qnet.train(batch_s, batch_a, batch_t, batch_n, batch_r)


    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):                                           
            actions_onehot[i][int(actions[i])] = 1      
        return actions_onehot   

    def mergeStateMatrices(self, stateMatrices):
        """ Merge state matrices to one state tensor """
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        total = np.zeros((7, 7))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 6
        return total

    def getStateMatrices(self, state):
        """ Return wall, ghosts, food, capsules matrices """ 
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.layout.walls
            matrix = np.zeros((height, width), dtype=np.int8)
            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell
            return matrix

        def getPacmanMatrix(state):
            """ Return matrix with pacman coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.food
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell

            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            capsules = state.data.layout.capsules
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1-i[1], i[0]] = 1

            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height 
        width, height = self.params['width'], self.params['height']
        observation = np.zeros((6, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)

        observation = np.swapaxes(observation, 0, 2)

        return observation

    def registerInitialState(self, state): # inspects the starting state

        # Check if transition_mode is in the state's options and use it for save_file
        if hasattr(state, 'data') and hasattr(state.data, 'options') and 'transition_mode' in state.data.options:
            self.transition_mode = state.data.options['transition_mode']
            self.params['save_file'] = self.transition_mode
            print(f"Setting save_file based on transition mode: {self.transition_mode}")
        else:
            print("Warning: No transition_mode found in state options, using default")
            
        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0
        self.last_min_food_distance = float('inf')  # Initialize food distance tracking

        # Reset state
        self.last_state = None
        self.current_state = self.getStateMatrices(state)

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.numeps += 1
        
        # Reset episode metrics for tracking
        self.episode_q_min = float('inf')
        self.episode_q_max = float('-inf')
        self.episode_q_sum = 0
        self.episode_q_count = 0
        self.ep_steps = 0

    def getAction(self, state):
        move = self.getMove(state)

        # Stop moving when not legal
        legal = state.getLegalActions(0)
        if move not in legal:
            move = Directions.STOP

        return move
