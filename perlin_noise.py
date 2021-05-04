import random as rn
import math as mt
import numpy as np
from numba import njit, jit



"Small mathsy functions"

@njit(cache = True, fastmath = True)
def random_unit_vector():
    angle = rn.random()*2*np.pi
    x = np.cos(angle)
    y = np.sin(angle)
    return x, y

@njit(cache = True)
def corners(x, y):
    l = mt.floor(x)
    r = mt.ceil(x)
    b = mt.floor(y)
    t = mt.ceil(y)
    return l, r, b, t

@njit(cache = True, fastmath= True)
def ease_curve(x):
    value = 0
    if 0 < x < 1:
        value = 6*x**5 - 15*x**4 + 10*x**3
    else:
        value = 1.0
    return value

@njit(cache = True)
def map_value(x, in_min, in_max, out_min, out_max):
    return((x - in_min)/(in_max - in_min))*(out_max - out_min) + out_min

# combine this with the falloff function, that should be faster
@njit
def gaussian_2d(x, y, x0, y0, sigma):
    x_diff = x - x0
    y_diff = y - y0
    return np.exp(-x_diff*x_diff/2*sigma*sigma - y_diff*y_diff/2*sigma*sigma)



"Core functions for image generation"

@njit(cache = True)
def gen_grid(grid_width, grid_height, tiling_NS = False, tiling_EW = False):
    grid_width, grid_height = grid_width + 1, grid_height + 1
    grid = np.empty((grid_width, grid_height, 2))
    for x in np.arange(grid_width):
        for y in np.arange(grid_height):
            grid[x][y][0], grid[x][y][1] = random_unit_vector()
    if tiling_NS is True:
        grid[grid_width - 1, :, :] = grid[0, :, :]
    if tiling_EW is True:
        grid[:, grid_height - 1, :] = grid[:, 0, :]
    return grid

# This is a bit over complicated, uses lots of memory, try to do it with lists or tuples
@njit(cache = True)
def gen_grids(grid_width, grid_height, octaves, grid_scale = 1, freq_scale = 2, tiling_NS = False, tiling_EW = False):
    width, height = grid_width*grid_scale + 1, grid_height*grid_scale + 1
    grids = np.zeros((octaves,
                      int(width*freq_scale**(octaves - 1) + 1),
                      int(height*freq_scale**(octaves - 1) + 1),
                      2))
    for octave in np.arange(octaves):
        for x in np.arange(width*freq_scale**octave):
            for y in np.arange(height*freq_scale**octave):
                grids[octave][x][y][0], grids[octave][x][y][1] = random_unit_vector()
        grids[octave][-1][0][0], grids[octave][0][-1][0] = width*freq_scale**octave, height*freq_scale**octave
        if tiling_NS is True:
            grids[octave, width*freq_scale**octave - 1, :, :] = grids[octave, 0, :, :]
        if tiling_EW is True:
            grids[octave, :, height*freq_scale**octave - 1, :] = grids[octave, :, 0, :]
    return grids

@njit(cache = True)
def value(x, y, grid):
    l, r, b, t = corners(x, y)
    dot_list = (grid[l][t][0]*(x - l) + grid[l][t][1]*(y - t),
                grid[r][t][0]*(x - r) + grid[r][t][1]*(y - t),
                grid[l][b][0]*(x - l) + grid[l][b][1]*(y - b),
                grid[r][b][0]*(x - r) + grid[r][b][1]*(y - b))
    weight_x = ease_curve(x - l)
    avg_t = dot_list[0] + weight_x*(dot_list[1] - dot_list[0])
    avg_b = dot_list[2] + weight_x*(dot_list[3] - dot_list[2])
    weight_y = ease_curve(y - b)
    point_value = avg_b + weight_y*(avg_t - avg_b)
    return point_value

@njit(cache = True)
def image(width, height, grid):
    grid_width, grid_height = np.shape(grid)[0] - 1, np.shape(grid)[1] - 1
    image = np.empty((width, height))
    for x in np.arange(width):
        for y in np.arange(height):
            image[x][y] = value((x*grid_width)/width, (y*grid_height)/height, grid)
    return image

@njit(cache = True)
def image_from_images(width, height, grid, image_x, image_y):
    image = np.empty((width, height))
    for x in np.arange(width):
        for y in np.arange(height):
            image[x][y] = value(image_x[x][y], image_y[x][y], grid)
    return image

# Doesn't really work yet, not sure why
@njit(cache = True)
def image_octaves(width, height, grids):
    image = np.empty((width, height))
    for octave in np.arange(len(grids)):
        grid_width, grid_height = grids[octave][-1][0][0], grids[octave][0][-1][0]
        for x in np.arange(width):
            for y in np.arange(height):
                image[x][y] = value((x*grid_width)/width, (y*grid_height)/height, grids[octave])
    return image

@njit(cache = True)
def image_bigpix(width, height, grid, pix_size):
    grid_width, grid_height = np.shape(grid)[0] - 1, np.shape(grid)[1] - 1
    image = np.empty((width, height))
    for x in np.arange(0, width, pix_size):
        for y in np.arange(0, height, pix_size):
            image[x : x + pix_size, y : y + pix_size] = value((x*grid_width)/width, (y*grid_height)/height, grid)
    return image



"Functions to apply to the image"

@njit(cache = True)
def normalise(image):
    normalised_image = (image - np.min(image))/np.ptp(image)
    return normalised_image

