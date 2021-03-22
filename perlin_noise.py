# https://eev.ee/blog/2016/05/29/perlin-noise/
# https://rtouti.github.io/graphics/perlin-noise-algorithm
# https://mzucker.github.io/html/perlin-noise-math-faq.html

import math as mt
import numpy as np


class perlin_noise:
    def __init__(self, width, height):
        self.size = [width, height]
        self.grid = self.vector_grid()


    def random_unit_vector(self):
        angle = np.random.random()*2*np.pi
        x = np.cos(angle)
        y = np.sin(angle)
        return [x, y]


    def vector_grid(self):
        grid_rows = self.size[0] + 1
        grid_columns = self.size[1] + 1
        grid = [[] for row in range(grid_rows)]
        for row in range(grid_rows):
            for column in range(grid_columns):
                grid[row].append(self.random_unit_vector())  
        return grid
    
    
    def ease_curve(self, p):
        # return 3*p**2 - 2*p**3
        if p <= 0:
            value = 0
        elif 0 < p < 1:
            value = 6*p**5 - 15*p**4 + 10*p**3
        elif 1 <= p:
            value = 1
        return value
    
    
    def value(self, x, y, full_output = False):
        # Define the corners of the current grid square.
        x0 = mt.floor(x)
        x1 = mt.ceil(x)
        y0 = mt.floor(y)
        y1 = mt.ceil(y)
    
        # Fetch the random unit vectors corresponding to the corners of the grid
        # square.
        v_ran00 = [self.grid[x0][y0][0], self.grid[x0][y0][1]]
        v_ran10 = [self.grid[x1][y0][0], self.grid[x1][y0][1]]
        v_ran01 = [self.grid[x0][y1][0], self.grid[x0][y1][1]]
        v_ran11 = [self.grid[x1][y1][0], self.grid[x1][y1][1]]
        
        # Define the vectors pointing from the corners to the coordinate (x, y).
        v_in00 = [x - x0, y - y0]
        v_in10 = [x - x1, y - y0]
        v_in01 = [x - x0, y - y1]
        v_in11 = [x - x1, y - y1]
        
        # Calculate the dot product of each of the random vectors with the inward
        # pointing vector originating from its corner.
        dot00 = v_ran00[0]*v_in00[0] + v_ran00[1]*v_in00[1]
        dot10 = v_ran10[0]*v_in10[0] + v_ran10[1]*v_in10[1]
        dot01 = v_ran01[0]*v_in01[0] + v_ran01[1]*v_in01[1]
        dot11 = v_ran11[0]*v_in11[0] + v_ran11[1]*v_in11[1]
        
        # Calculate the value of the point (x, y)
        weight_x = self.ease_curve(x - x0)
        avg_upper = dot00 + weight_x*(dot10 - dot00)
        avg_lower = dot01 + weight_x*(dot11 - dot01) # Why are these this way round? It's all upside down for some reason
        weight_y = self.ease_curve(y - y0)
        point_value = avg_upper + weight_y*(avg_lower - avg_upper)
        # point_value = self.ease_curve(point_value) # This produces a cool effect!
        
        if full_output is False:
            return point_value
        
        elif full_output is True:
            output = [[[x0, y0], v_ran00, v_in00, dot00],
                      [[x1, y0], v_ran10, v_in10, dot10],
                      [[x0, y1], v_ran01, v_in01, dot01],
                      [[x1, y1], v_ran11, v_in11, dot11]]
            return point_value, output


    def image(self,  width, height):
        # Make an array for the image.
        image = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                # The self.size[0]/width bit scales the index x down to a value in the noise map.
                point_value = self.value(x*(self.size[0]/width), y*(self.size[1]/height))
                image[y][x] = point_value
        return image
    
    def line(self, points):
        x = [n for n in range(points)]
        y = [self.value(0, n*self.size[1]/points) for n in range(points)]
        return x, y