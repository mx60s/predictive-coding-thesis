import malmoenv
import math
import gym
import heapq
import random
from typing import Tuple, List

class AStarAgent:
    """
    An agent which comes up with random paths throughout the environment, and then generates the 
    actions via A* to traverse them.
    """
    def __init__(self, start_pos: Tuple, bounds: Tuple, obstacles: List):
        self.x, self.y, self.z = start_pos
        self.x_bound = bounds[0]
        self.z_bound = bounds[1]
        self.target = None
        self.obstacles = obstacles

    def _is_obstacle(self, x, y, z):
        x_f = math.floor(self.x)
        z_f = math.floor(self.z)
        for o in self.obstacles:
            if x_f == o[0] and z_f == o[1]:
                #print('obstacle at', o)
                return True
        return False

    def _generate_target(self):
        while True:
            # not sure about the addition with my bounds
            # tempted to just floor everything in the agent
            x = random.randint(self.x_bound[0], self.x_bound[1]) + 0.5
            z = random.randint(self.z_bound[0], self.z_bound[1]) + 0.5
            dist = abs(x - self.x) + abs(z - self.z)
            #print('dist', dist)
            if dist >= 10 and not self._is_obstacle(x, self.y, z):
                self.target = (x, self.y, z)
                #print('new target:', self.target)
                break
    
    def _heuristic(self, a, b):
        return (abs(a[0] - b[0]) + abs(a[2] - b[2])) 
    
    def _get_neighbors(self, point):
        x, y, z = point
        neighbors = [(x+1, y, z), (x-1, y, z), (x, y, z+1), (x, y, z-1)]
        return [n for n in neighbors if not self._is_obstacle(*n)]
   
    def _a_star_search(self):
        """Find the next step towards the target using the A* algorithm."""
        start = (self.x, self.y, self.z)
        target = self.target
        closed_set = set()
        open_set = [(0, start)]
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, target)}
        came_from = {}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == target:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path[0]  # Return the next step after the start

            #open_set.remove(current)
            closed_set.add(current)

            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set or self._is_obstacle(*neighbor):
                    continue  # Ignore the neighbor which is already evaluated.
                #if neighbor not in open_set:  # Discover a new node
                #    open_set.add(neighbor)
                #    heapq.heappush(open_heap, (f_score.get(neighbor, float('inf')), neighbor))

                # The distance from start to a neighbor
                tentative_g_score = g_score[current] + 1  # Each step costs 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one. Record it!
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, target)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # If there is no path to the target


    def next_step(self):
        if not self.target or (self.x, self.y, self.z) == self.target:
            #print('generating new target')
            self._generate_target()
          
        dist_target= self._heuristic((self.x, self.y, self.z), self.target)
        #print(dist_target)
        if dist_target < 7:
            #print('close enough, generating new target')
            self._generate_target()
    
        next_point = self._a_star_search()
        
        # Translate point to action
        dx = next_point[0] - self.x
        dz = next_point[2] - self.z

        #print('dx dz', dx, dz)

        if dx == 1:
            return 3
        elif dx == -1:
            return 2
        elif dz == 1:
            return 0
        elif dz == -1:
            return 1

