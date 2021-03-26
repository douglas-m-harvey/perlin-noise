# https://eev.ee/blog/2016/05/29/perlin-noise/
# https://rtouti.github.io/graphics/perlin-noise-algorithm
# https://mzucker.github.io/html/perlin-noise-math-faq.html
# https://web.archive.org/web/20160530124230/http://freespace.virgin.net/hugo.elias/models/m_perlin.htm


import math as mt
import numpy as np
import time


def random_unit_vector():
    # Generate a random unit vector.
    angle = np.random.random()*2*np.pi
    x = np.cos(angle)
    y = np.sin(angle)
    return [x, y]

def map_value(x, in_min, in_max, out_min, out_max):
    # Map the value x in the interval [in_min, in_max] linearly to the interval [out_min, out_max]
    return((x - in_min)/(in_max - in_min))*(out_max - out_min) + out_min


class perlin_noise:
    def __init__(self, octaves = 1, octave_scale = 0.5):
        # Define a list of dictionaries, one for each octave.
        self.grid = [{} for n in range(octaves)]
        self.octaves = octaves
        self.octave_scale = octave_scale


    def ease_curve(self, p):
        # Define an s-shaped curve to smooth the boundaries between grid squares (I think).
        if p <= 0:
            value = 0
        elif 0 < p < 1:
            value = 6*p**5 - 15*p**4 + 10*p**3
        elif 1 <= p:
            value = 1
        # return 3*p**2 - 2*p**3 # Different ease curve, not sure if there's really any difference.
        return value


    def value(self, x, y, full_output = False):
        point_value = 0
        
        for octave in range(self.octaves):
            # Take the input coordinate in the interval [0, 1] (What happens if it's outside this?) and
            # map it to the interval [0, octave_scale*octave + 1]. As octave increases, the spacing of 
            # the output coordinates increases, so they will be spread out over more grid squares. This
            # grid is then 'shrunk' down to the size of the originl grid to give higher frequency noise.
            x = map_value(x, 0, 1, 0, (self.octave_scale*octave + 1))
            y = map_value(y, 0, 1, 0, (self.octave_scale*octave + 1))
            
            # Use the scaled coordinates to determine the corners of the current grid square
            x0 = mt.floor(x)
            x1 = mt.ceil(x)
            y0 = mt.floor(y)
            y1 = mt.ceil(y)
            
            # Check if the corner's coordinates are already in the dictionary for this octave and, if not,
            # add them as a key with a new random unit vector as the corresponding value. Then assign the
            # corner's vector to a pxy variable.
            if (x0, y0) not in self.grid[octave]:
                vector = random_unit_vector()
                self.grid[octave].update({(x0, y0) : vector})
                p00 = vector
            elif (x0, y0) in self.grid[octave]:
                p00 = self.grid[octave][(x0, y0)]
    
            if (x1, y0) not in self.grid[octave]:
                vector = random_unit_vector()
                self.grid[octave].update({(x1, y0) : vector})
                p10 = vector
            elif (x1, y0) in self.grid[octave]:
                p10 = self.grid[octave][(x1, y0)]
    
            if (x0, y1) not in self.grid[octave]:
                vector = random_unit_vector()
                self.grid[octave].update({(x0, y1) : vector})
                p01 = vector
            elif (x0, y1) in self.grid[octave]:
                p01 = self.grid[octave][(x0, y1)]
    
            if (x1, y1) not in self.grid[octave]:
                vector = random_unit_vector()
                self.grid[octave].update({(x1, y1) : vector})
                p11 = vector
            elif (x1, y1) in self.grid[octave]:
                p11 = self.grid[octave][(x1, y1)]            
            
            # Define the vectors pointing from the corners to the coordinate (x, y).
            v_in00 = [x - x0, y - y0]
            v_in10 = [x - x1, y - y0]
            v_in01 = [x - x0, y - y1]
            v_in11 = [x - x1, y - y1]
            
            # For each corner, calculate the dot product of the random unit vector with the inward pointing
            # vector.
            dot00 = p00[0]*v_in00[0] + p00[1]*v_in00[1]
            dot10 = p10[0]*v_in10[0] + p10[1]*v_in10[1]
            dot01 = p01[0]*v_in01[0] + p01[1]*v_in01[1]
            dot11 = p11[0]*v_in11[0] + p11[1]*v_in11[1]
            
            # Calculate the value of the point (x, y) by taking a weighted average along the top and bottom
            # edges, then taking a weighted average of these two averages... or something like that.
            weight_x = self.ease_curve(x - x0)
            avg_upper = dot00 + weight_x*(dot10 - dot00)
            avg_lower = dot01 + weight_x*(dot11 - dot01) # Why are these this way round? It's all upside down for some reason
            weight_y = self.ease_curve(y - y0)
            # Divide the final value by 2**octave so as the noise frequency increases, amplitude decreases.
            point_value += (avg_upper + weight_y*(avg_lower - avg_upper))/2**octave
            # point_value = self.ease_curve(point_value) # This produces a cool effect!
        
        if full_output is False:
            return point_value
        
        elif full_output is True:
            # Output some other bits and pieces if that's what you're into.
            output = [[[x0, y0], p00, v_in00, dot00],
                      [[x1, y0], p10, v_in10, dot10],
                      [[x0, y1], p01, v_in01, dot01],
                      [[x1, y1], p11, v_in11, dot11]] 
            return point_value, output


    def image(self,  width, height, scale = 1, normalised = True, lower_lim = 0, upper_lim = 1, show_progress = False):
        # Make an array for the image.
        image = np.zeros((height, width))
        # Take the maximum side length.
        side_length = max([height, width])
        # Set up some timing stuff and readout headings.
        if show_progress is True:
            progress = 0
            duration = 0
            print("Progress:\tStep time:")
        for y_pixel in range(height):
            for x_pixel in range(width):
                # Start timer.
                if show_progress is True:
                    start = time.perf_counter()
                # Map the input x and y coordinates to the interval [0, scale] to control "zoom".
                # The side_length bit keeps the noise grid square so the image doesn't look stretched.
                x_coord = map_value(x_pixel, 0, side_length, 0, scale)
                y_coord = map_value(y_pixel, 0, side_length, 0, scale)
                image[y_pixel][x_pixel] = self.value(x_coord, y_coord)
                # Print current progress percent and time taken to generate the last percent of the image.
                if show_progress is True:
                    progress_temp = np.uint8(round((x_pixel + y_pixel*width)/(width*height), 2)*100)
                    # End timer.
                    end = time.perf_counter()
                    duration += (end - start)
                    if progress_temp > progress:
                        progress = progress_temp
                        print(str(progress) + "%\t\t\t" + str(round(duration, 3)) + "s")
                        duration = 0
        if normalised is True:
            # After the image is generated, map its values to the interval [lower_lim, upper_lim].
            max_val = np.max(image)
            min_val = np.min(image)
            for y_pixel in range(height):
                for x_pixel in range(width):
                    value = image[y_pixel][x_pixel]
                    value = map_value(value, min_val, max_val, lower_lim, upper_lim)
                    image[y_pixel][x_pixel] = value
        return image


    def line(self, no_points, scale = 1, normalised = True, lower_lim = 0, upper_lim = 1, show_progress = False):
        # Make the list of x-coordinates and an empty list for the y-coordinates.
        x_values = [x_value for x_value in range(no_points)]
        y_values = []
        if show_progress is True:
            progress = 0
            duration = 0
            print("Progress:\tStep time:")
        for point in range(no_points):
            if show_progress is True:
                start = time.perf_counter()
            # It's pretty much the same as the image function from here on out, just in 1 dimension.
            y_value = map_value(point, 0, no_points, 0, scale)
            y_values.append(self.value(y_value, 0))
            if show_progress is True:
                progress_temp = np.uint8(round(point/no_points, 2)*100)
                end = time.perf_counter()
                duration += (end - start)
                if progress_temp > progress:
                    progress = progress_temp
                    print(str(progress) + "%\t\t\t" + str(round(duration, 3)) + "s")
                    duration = 0
        if normalised is True:
            max_val = max(y_values)
            min_val = min(y_values)
            for point in range(no_points):
                value = y_values[point]
                value = map_value(value, min_val, max_val, lower_lim, upper_lim)
                y_values[point] = value
        return x_values, y_values
