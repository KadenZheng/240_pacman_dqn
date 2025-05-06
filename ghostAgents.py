# ghostAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util
import math


class GhostAgent(Agent):
    """base ghost agent class"""
    def __init__(self, index):
        self.index = index

    def getAction(self, state):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):
        """returns a counter for action distribution"""
        util.raiseNotDefined()


class RandomGhost(GhostAgent):
    """provides unpredictability through uniform random selection"""

    def getDistribution(self, state):
        dist = util.Counter()
        for a in state.getLegalActions(self.index):
            dist[a] = 1.0
        dist.normalize()
        return dist


class DirectionalGhost(GhostAgent):
    """supports basic chasing/fleeing behavior for more challenge"""

    def __init__(self, index, prob_attack=0.8, prob_scaredFlee=0.8):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution(self, state):
        # read variables from state
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared:
            speed = 0.5

        actionVectors = [Actions.directionToVector(
            a, speed) for a in legalActions]
        newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # select best actions given the state
        distancesToPacman = [manhattanDistance(
            pos, pacmanPosition) for pos in newPositions]
        if isScared:
            bestScore = max(distancesToPacman)
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min(distancesToPacman)
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip(
            legalActions, distancesToPacman) if distance == bestScore]

        # construct distribution
        dist = util.Counter()
        for a in bestActions:
            dist[a] = bestProb / len(bestActions)
        for a in legalActions:
            dist[a] += (1 - bestProb) / len(legalActions)
        dist.normalize()
        return dist


class HighlyRandomGhost(GhostAgent):
    """makes gameplay less predictable through increased randomness"""
    
    def __init__(self, index, randomness=0.7):
        self.index = index
        self.randomness = randomness  # probability of taking a random action
    
    def getDistribution(self, state):
        # get legal actions
        legalActions = state.getLegalActions(self.index)
        
        # create uniform distribution for random movement
        dist = util.Counter()
        for a in legalActions:
            dist[a] = 1.0
        
        # with probability (1-randomness), prefer moving towards pacman or away if scared
        if random.random() > self.randomness:
            ghostState = state.getGhostState(self.index)
            pos = state.getGhostPosition(self.index)
            isScared = ghostState.scaredTimer > 0
            pacmanPosition = state.getPacmanPosition()
            
            # calculate new positions
            speed = 0.5 if isScared else 1
            actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
            newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]
            
            # calculate distances to pacman
            distancesToPacman = [manhattanDistance(pos, pacmanPosition) for pos in newPositions]
            
            # select best actions based on scared state
            if isScared:
                bestScore = max(distancesToPacman)  # flee
            else:
                bestScore = min(distancesToPacman)  # chase
                
            bestActions = [action for action, distance in zip(legalActions, distancesToPacman) 
                          if distance == bestScore]
            
            # increase weight for best actions
            for a in bestActions:
                dist[a] += 2.0
        
        # normalize the distribution
        dist.normalize()
        return dist


class AggressiveGhost(GhostAgent):
    """provides higher difficulty through more determined pursuit"""
    
    def __init__(self, index, prob_attack=0.95):
        self.index = index
        self.prob_attack = prob_attack  # higher than standard directional ghost
    
    def getDistribution(self, state):
        # read variables from state
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0
        
        # if scared, behavior similar to directional ghost
        if isScared:
            return self._getScaredDistribution(state, ghostState, legalActions, pos)
        
        # not scared - prioritize chase mode with high probability
        pacmanPosition = state.getPacmanPosition()
        
        # calculate potential new positions
        actionVectors = [Actions.directionToVector(a, 1) for a in legalActions]
        newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]
        
        # calculate distances to pacman
        distancesToPacman = [manhattanDistance(new_pos, pacmanPosition) for new_pos in newPositions]
        bestScore = min(distancesToPacman)
        
        # identify actions that lead closest to pacman
        bestActions = [action for action, distance in zip(legalActions, distancesToPacman) 
                      if distance == bestScore]
        
        # construct distribution with high weight for chase actions
        dist = util.Counter()
        for a in bestActions:
            dist[a] = self.prob_attack / len(bestActions)
        for a in legalActions:
            dist[a] += (1 - self.prob_attack) / len(legalActions)
        
        dist.normalize()
        return dist
    
    def _getScaredDistribution(self, state, ghostState, legalActions, pos):
        """handles scared state to ensure appropriate fleeing"""
        pacmanPosition = state.getPacmanPosition()
        actionVectors = [Actions.directionToVector(a, 0.5) for a in legalActions]
        newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]
        
        # when scared, try to flee from pacman
        distancesToPacman = [manhattanDistance(new_pos, pacmanPosition) for new_pos in newPositions]
        bestScore = max(distancesToPacman)
        prob_scaredFlee = 0.8  # default flee probability
        
        bestActions = [action for action, distance in zip(legalActions, distancesToPacman) 
                      if distance == bestScore]
        
        # construct distribution
        dist = util.Counter()
        for a in bestActions:
            dist[a] = prob_scaredFlee / len(bestActions)
        for a in legalActions:
            dist[a] += (1 - prob_scaredFlee) / len(legalActions)
        
        dist.normalize()
        return dist


class PathfindingGhost(GhostAgent):
    """improves ghost intelligence through better target tracking"""
    
    def __init__(self, index, heuristic_weight=0.8, path_randomness=0.3):
        self.index = index
        self.heuristic_weight = heuristic_weight  # balances path optimality
        self.path_randomness = path_randomness    # prevents predictable patterns
    
    def getDistribution(self, state):
        # read variables from state
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0
        pacmanPosition = state.getPacmanPosition()
        
        # if scared, behave more like a standard ghost
        if isScared:
            return self._getScaredDistribution(state, ghostState, legalActions, pos, pacmanPosition)
        
        # a* pathfinding for chasing pacman
        actionScores = {}
        for action in legalActions:
            # calculate new position after action
            vector = Actions.directionToVector(action, 1)
            new_pos = (pos[0] + vector[0], pos[1] + vector[1])
            
            # calculate g-score (distance from current position) - always 1 for adjacent positions
            g_score = 1
            
            # calculate h-score (heuristic distance to target)
            h_score = manhattanDistance(new_pos, pacmanPosition)
            
            # calculate f-score (estimated total cost)
            f_score = g_score + self.heuristic_weight * h_score
            
            # lower score is better (closer to pacman)
            actionScores[action] = f_score
        
        # find best actions based on a* scores
        min_score = min(actionScores.values())
        bestActions = [action for action, score in actionScores.items() 
                      if score == min_score]
        
        # calculate scores for second-best actions (for probabilistic behavior)
        secondBestActions = [action for action, score in actionScores.items() 
                           if score > min_score]
        
        # construct distribution with a* pathfinding and randomness
        dist = util.Counter()
        
        # assign high probability to best actions
        for a in bestActions:
            dist[a] = (1 - self.path_randomness) / len(bestActions)
        
        # assign some probability to other actions (exploration)
        for a in secondBestActions:
            dist[a] = self.path_randomness / max(1, len(secondBestActions))
        
        dist.normalize()
        return dist
    
    def _getScaredDistribution(self, state, ghostState, legalActions, pos, pacmanPosition):
        """enables appropriate fleeing behavior when vulnerable"""
        actionScores = {}
        for action in legalActions:
            # calculate new position after action
            vector = Actions.directionToVector(action, 0.5)  # slower when scared
            new_pos = (pos[0] + vector[0], pos[1] + vector[1])
            
            # for scared ghosts, we want to maximize distance from pacman
            distance = manhattanDistance(new_pos, pacmanPosition)
            actionScores[action] = distance  # higher score is better (further from pacman)
        
        # find best actions (furthest from pacman)
        max_score = max(actionScores.values())
        bestActions = [action for action, score in actionScores.items() 
                      if score == max_score]
        
        # construct distribution
        dist = util.Counter()
        prob_flee = 0.8
        
        for a in bestActions:
            dist[a] = prob_flee / len(bestActions)
        for a in legalActions:
            dist[a] += (1 - prob_flee) / len(legalActions)
        
        dist.normalize()
        return dist


class TransitionFunctionController:
    """enables ghost behavior adaptation for training or testing purposes"""
    
    GHOST_TYPES = {
        'random': RandomGhost,
        'directional': DirectionalGhost,
        'highly_random': HighlyRandomGhost,
        'aggressive': AggressiveGhost,
        'pathfinding': PathfindingGhost
    }
    
    def __init__(self, mode='fixed', transition_steps=5000):
        """
        initialize the transition function controller.
        
        args:
            mode: determines behavior adaptation strategy
            transition_steps: controls progression speed
        """
        self.mode = mode
        self.transition_steps = transition_steps
        self.current_step = 0
        self.current_ghost_type = 'directional'  # default ghost type
        self.ghost_params = {}  # parameters for the current ghost
        
        # default parameters for each ghost type
        self.default_params = {
            'directional': {'prob_attack': 0.8, 'prob_scaredFlee': 0.8},
            'highly_random': {'randomness': 0.7},
            'aggressive': {'prob_attack': 0.95},
            'pathfinding': {'heuristic_weight': 0.8, 'path_randomness': 0.3}
        }
        
        # set initial parameters
        self.reset_ghost_params()
    
    def reset_ghost_params(self):
        """restores default settings for consistency"""
        if self.current_ghost_type in self.default_params:
            self.ghost_params = self.default_params[self.current_ghost_type].copy()
        else:
            self.ghost_params = {}
    
    def update(self, episode_num):
        """
        modifies ghost behavior based on training progress
        
        args:
            episode_num: current training episode
        
        returns:
            dict: updated ghost parameters
        """
        self.current_step = episode_num
        
        if self.mode == 'fixed':
            # no change in fixed mode
            pass
        
        elif self.mode == 'progressive':
            # progressively shift between ghost types
            progress = min(1.0, self.current_step / self.transition_steps)
            
            if progress < 0.25:
                self.current_ghost_type = 'directional'
            elif progress < 0.5:
                self.current_ghost_type = 'highly_random'
            elif progress < 0.75:
                self.current_ghost_type = 'aggressive'
            else:
                self.current_ghost_type = 'pathfinding'
            
            self.reset_ghost_params()
        
        elif self.mode == 'random':
            # randomly select a ghost type each episode
            ghost_types = list(self.GHOST_TYPES.keys())
            self.current_ghost_type = random.choice(ghost_types)
            self.reset_ghost_params()
        
        elif self.mode == 'domain_random':
            # randomize ghost parameters while keeping the same type
            if episode_num % 100 == 0:  # change type every 100 episodes
                ghost_types = list(self.GHOST_TYPES.keys())
                self.current_ghost_type = random.choice(ghost_types)
            
            # randomize parameters
            if self.current_ghost_type == 'directional':
                self.ghost_params = {
                    'prob_attack': random.uniform(0.6, 0.9),
                    'prob_scaredFlee': random.uniform(0.6, 0.9)
                }
            elif self.current_ghost_type == 'highly_random':
                self.ghost_params = {
                    'randomness': random.uniform(0.5, 0.9)
                }
            elif self.current_ghost_type == 'aggressive':
                self.ghost_params = {
                    'prob_attack': random.uniform(0.85, 0.98)
                }
            elif self.current_ghost_type == 'pathfinding':
                self.ghost_params = {
                    'heuristic_weight': random.uniform(0.6, 0.95),
                    'path_randomness': random.uniform(0.1, 0.5)
                }
        
        return {
            'ghost_type': self.current_ghost_type,
            'params': self.ghost_params
        }
    
    def create_ghost(self, ghost_index):
        """
        instantiates appropriate ghost type for current settings
        
        args:
            ghost_index: ghost's position in game
        
        returns:
            ghostagent: configured ghost instance
        """
        ghost_class = self.GHOST_TYPES.get(self.current_ghost_type, DirectionalGhost)
        
        if self.current_ghost_type == 'random':
            return ghost_class(ghost_index)
        
        return ghost_class(ghost_index, **self.ghost_params)
