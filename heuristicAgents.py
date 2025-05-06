#!/usr/bin/env python3
"""
HeuristicAgent.py
----------------
Provides a heuristic-based Pac-Man agent that uses shortest-path planning
to navigate the maze, collect food, and avoid ghosts.

This agent serves as a baseline for comparison with RL agents.
"""

from game import Agent
from game import Directions
import random
import util
import math
from util import manhattanDistance

class HeuristicAgent(Agent):
    """
    A heuristic agent that uses path planning to navigate the Pac-Man maze.
    The agent tries to:
    1. Eat nearby food
    2. Avoid ghosts (unless they are scared)
    3. Chase scared ghosts when safe
    4. Use capsules strategically
    """
    
    def __init__(self):
        """Initialize the heuristic agent"""
        self.last_action = None
        self.actions_taken = 0
        
        # Safety parameters
        self.ghost_danger_distance = 3  # How close ghosts get before we consider them dangerous
        self.ghost_chase_distance = 5   # How close scared ghosts should be for us to chase them
        self.capsule_distance = 10      # How close capsules should be for us to prioritize them
        
        # Exploration parameters
        self.exploration_rate = 0.1     # Probability of taking a random move
        self.stuck_threshold = 5        # Number of repeated positions before considering "stuck"
        self.recent_positions = []      # Track recent positions to detect being stuck
        
        # Performance monitoring
        self.food_eaten = 0
        self.total_food = 0
        self.score = 0
    
    def registerInitialState(self, state):
        """
        Initialize agent state from the initial game state
        """
        self.total_food = state.getNumFood()
        self.food_eaten = 0
        self.score = state.getScore()
        self.recent_positions = []
        
        # Pre-compute all shortest paths
        self.walls = state.getWalls()
        self.shortest_paths = {}
    
    def getAction(self, state):
        """
        Get the action using heuristic-based decision making
        """
        # Update agent state
        self.actions_taken += 1
        current_pos = state.getPacmanPosition()
        legal_actions = state.getLegalPacmanActions()
        
        # Remove STOP action if present
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)
        
        # If no legal actions, return STOP
        if not legal_actions:
            return Directions.STOP
        
        # Track recent positions to detect being stuck
        self.recent_positions.append(current_pos)
        if len(self.recent_positions) > 10:
            self.recent_positions.pop(0)
        
        # Check if stuck in a loop
        is_stuck = False
        if len(self.recent_positions) >= self.stuck_threshold:
            # Count frequency of current position
            position_count = self.recent_positions.count(current_pos)
            if position_count >= self.stuck_threshold // 2:
                is_stuck = True
        
        # Random exploration to break out of loops
        if is_stuck or random.random() < self.exploration_rate:
            return random.choice(legal_actions)
        
        # Get the current state of the game
        food = state.getFood()
        capsules = state.getCapsules()
        ghost_states = state.getGhostStates()
        scared_time = min([ghost.scaredTimer for ghost in ghost_states]) if ghost_states else 0
        
        # Calculate action scores based on various factors
        action_scores = {action: 0 for action in legal_actions}
        
        # Factor 1: Distance to nearest food
        food_action_scores = self._evaluateFoodActions(state, legal_actions, current_pos, food)
        for action, score in food_action_scores.items():
            action_scores[action] += score
        
        # Factor 2: Ghost avoidance/pursuit
        ghost_action_scores = self._evaluateGhostActions(state, legal_actions, current_pos, ghost_states)
        for action, score in ghost_action_scores.items():
            action_scores[action] += score
        
        # Factor 3: Capsule collection
        if capsules and scared_time < 5:  # Only go for capsules if ghosts aren't already scared
            capsule_action_scores = self._evaluateCapsuleActions(state, legal_actions, current_pos, capsules)
            for action, score in capsule_action_scores.items():
                action_scores[action] += score
        
        # Choose the action with the highest score
        best_score = max(action_scores.values())
        best_actions = [action for action, score in action_scores.items() if score == best_score]
        
        # Break ties randomly
        chosen_action = random.choice(best_actions)
        self.last_action = chosen_action
        
        return chosen_action
    
    def _evaluateFoodActions(self, state, legal_actions, current_pos, food):
        """
        Evaluate actions based on proximity to food
        Returns a dict mapping actions to scores
        """
        scores = {action: 0 for action in legal_actions}
        
        # Find all food positions
        food_positions = []
        for x in range(food.width):
            for y in range(food.height):
                if food[x][y]:
                    food_positions.append((x, y))
        
        if not food_positions:
            return scores
        
        # Find closest food
        closest_food = min(food_positions, key=lambda pos: manhattanDistance(current_pos, pos))
        
        # Score actions based on distance to closest food
        for action in legal_actions:
            # Get next position after taking this action
            next_pos = self._getNextPosition(current_pos, action)
            
            # Calculate distance from next position to closest food
            dist = manhattanDistance(next_pos, closest_food)
            
            # Higher score for actions that get us closer to food
            scores[action] = 10.0 / (dist + 1)
            
            # Bonus for actions that directly lead to food
            if next_pos in food_positions:
                scores[action] += 20
        
        return scores
    
    def _evaluateGhostActions(self, state, legal_actions, current_pos, ghost_states):
        """
        Evaluate actions based on proximity to ghosts
        Returns a dict mapping actions to scores
        """
        scores = {action: 0 for action in legal_actions}
        
        for ghost in ghost_states:
            ghost_pos = ghost.getPosition()
            is_scared = ghost.scaredTimer > 0
            
            for action in legal_actions:
                next_pos = self._getNextPosition(current_pos, action)
                dist = manhattanDistance(next_pos, ghost_pos)
                
                if is_scared:
                    # Chase scared ghosts if they're close enough
                    if dist <= self.ghost_chase_distance:
                        # Higher score for moving closer to scared ghosts
                        scores[action] += 15.0 / (dist + 1)
                else:
                    # Avoid regular ghosts
                    if dist <= self.ghost_danger_distance:
                        # Heavily penalize moving closer to ghosts
                        scores[action] -= 30.0 / (dist + 0.1)
        
        return scores
    
    def _evaluateCapsuleActions(self, state, legal_actions, current_pos, capsules):
        """
        Evaluate actions based on proximity to capsules
        Returns a dict mapping actions to scores
        """
        scores = {action: 0 for action in legal_actions}
        
        if not capsules:
            return scores
        
        # Find closest capsule
        closest_capsule = min(capsules, key=lambda pos: manhattanDistance(current_pos, pos))
        
        # Check if there are nearby ghosts that make capsules valuable
        ghost_states = state.getGhostStates()
        nearby_ghosts = False
        for ghost in ghost_states:
            if not ghost.scaredTimer > 0:
                ghost_dist = manhattanDistance(current_pos, ghost.getPosition())
                if ghost_dist < self.capsule_distance:
                    nearby_ghosts = True
                    break
        
        # If there are nearby ghosts, prioritize capsules
        capsule_priority = 15 if nearby_ghosts else 5
        
        for action in legal_actions:
            next_pos = self._getNextPosition(current_pos, action)
            dist = manhattanDistance(next_pos, closest_capsule)
            
            # Higher score for actions that get us closer to capsules
            scores[action] = capsule_priority / (dist + 1)
            
            # Bonus for actions that directly lead to capsules
            if next_pos == closest_capsule:
                scores[action] += 25
        
        return scores
    
    def _getNextPosition(self, current_pos, action):
        """
        Return the next position after taking an action from current_pos
        """
        x, y = current_pos
        
        if action == Directions.NORTH:
            return (x, y + 1)
        elif action == Directions.SOUTH:
            return (x, y - 1)
        elif action == Directions.EAST:
            return (x + 1, y)
        elif action == Directions.WEST:
            return (x - 1, y)
        
        return current_pos  # Default case for STOP
    
    def _getShortestPath(self, start, end):
        """
        Find the shortest path between start and end positions using BFS
        """
        # Check if we've already computed this path
        if (start, end) in self.shortest_paths:
            return self.shortest_paths[(start, end)]
        
        # Use BFS to find the shortest path
        fringe = util.Queue()
        fringe.push((start, []))
        visited = set()
        
        while not fringe.isEmpty():
            pos, path = fringe.pop()
            
            if pos == end:
                # Cache the result
                self.shortest_paths[(start, end)] = path
                return path
            
            if pos in visited:
                continue
                
            visited.add(pos)
            
            # Get neighbors
            x, y = pos
            neighbors = []
            if not self.walls[x+1][y]:
                neighbors.append(((x+1, y), Directions.EAST))
            if not self.walls[x-1][y]:
                neighbors.append(((x-1, y), Directions.WEST))
            if not self.walls[x][y+1]:
                neighbors.append(((x, y+1), Directions.NORTH))
            if not self.walls[x][y-1]:
                neighbors.append(((x, y-1), Directions.SOUTH))
            
            for neighbor, direction in neighbors:
                if neighbor not in visited:
                    fringe.push((neighbor, path + [direction]))
        
        # No path found
        self.shortest_paths[(start, end)] = []
        return []


def createHeuristicAgent(index=0):
    """
    Helper function to create a heuristic agent
    Can be used with command-line option "-p createHeuristicAgent"
    """
    return HeuristicAgent() 