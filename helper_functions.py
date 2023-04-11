# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 13:49:49 2023

@author: Sami
"""

import numpy as np
from scipy.special import binom
import random
from skimage.draw import polygon
import scipy.interpolate
import collections

def make_curves(curve, mask_original,curve_length,grid_size=3,direction=None):
    mask = mask_original.copy()
    if len(curve) == 0:
        # Getting the pixels available
        x, y = np.where(mask == 0)
        
        # Chossing a random pixel among the one availables
        ind = np.random.randint(len(x))
        first_elem = x[ind] + y[ind] * grid_size
        
        #We mask this pixel, and call the function again
        mask[first_elem % grid_size, first_elem // grid_size] = 1
        curve.append(first_elem)
        return make_curves(curve, mask,curve_length,grid_size=grid_size,direction=direction)
    else:
        # We look at the last pixel of the curve and get the coordinate of 
        # its neihbours
        x_end = curve[-1] % grid_size
        y_end = curve[-1] // grid_size

        consecutives_x = [x_end - 1, x_end + 1]
        consecutives_x = [i for i in consecutives_x if i >= 0 and i < grid_size]
        consecutives_y = [y_end - 1, y_end + 1]
        consecutives_y = [i for i in consecutives_y if i >= 0 and i < grid_size]

        neighbors = [(consecutives_x[i],y_end) for i in range(len(consecutives_x))]+[(x_end,consecutives_y[i]) for i in range(len(consecutives_y))]
        directions = [0 for i in range(len(consecutives_x))] + [1 for i in range(len(consecutives_y))] # vertical, horizontal
        
        
        while len(curve) < curve_length:  
            possible_next_value = []
            possible_next_value_directions = []
            # If a neighbouring pixel is not masked, we record it as a
            # a possible value
            for i in range(len(neighbors)):
                if mask[neighbors[i]] == 0:
                    possible_next_value.append(neighbors[i])
                    possible_next_value_directions.append(directions[i])
            
            # We randomly choose the next pixel among the available ones
            # and mask the nehbours 
            if direction is None:
                next_value = random.choice(possible_next_value)
            else:
                possible_next_value = np.array(possible_next_value)
                possible_next_value_directions = np.array(possible_next_value_directions)
                next_value = random.choice(possible_next_value[possible_next_value_directions == direction])
            curve.append(next_value[0] + next_value[1] * grid_size)
            for i in range(len(possible_next_value)):
                mask[possible_next_value[i][0], possible_next_value[i][1]] = 1
            return make_curves(curve, mask,curve_length,grid_size=grid_size,direction=direction)
        else:
            # If we reached the length of the curve, we only mask the nehbours
            for i in range(len(neighbors)):
                mask[neighbors[i]] = 1
            return curve, mask
        



bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)
    
def make_blob(grid_size):
    rad = 0.9999
    edgy = 0.9999
    n = 7
    c = [np.random.randint(30),np.random.randint(30)]
    a = get_random_points(n=n, scale=50) + c
    x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)   
    x_blob,y_blob = polygon(x, y, (grid_size,grid_size)) #filling in
    return x_blob,y_blob


def test_network(t,CurveLength,grid_size,TrialNumber,n,device,save_activities,only_blue,verbose=True):
    prev_exploitation_probability = n.exploitation_probability
    n.exploitation_probability = 1
    n.save_activities = save_activities
    corrects = []
    target_history = []
    distr_history = []
    display = []
    t.curve_length = CurveLength
    t.grid_size = grid_size
    t.only_blue = only_blue
    action = 0
    n.save_activities = save_activities
    if save_activities:
        n.saveXmod = [[]]
        n.saveY2mod = [[]]
        n.saveY3mod = [[]]
        n.saveY6mod = [[]]
        n.saveQ = [[]]
    for p in range(TrialNumber):
      trial_running = True
      display.append([])
      new_input, reward, trialEnd= t.do_step(action)
      while trial_running:
        action = n.do_step(new_input,reward,trialEnd,device)
        new_input, reward, trialEnd = t.do_step(action)
        display[p].append(new_input)
        if trialEnd:
          trial_running = False
          if reward == 0:
            corrects.append(0)
          else:
            corrects.append(1)
      target_history.append(t.target_curve)
      distr_history.append(t.distractor_curve) 
    if verbose:
        print(np.mean(corrects))
    n.exploitation_probability = prev_exploitation_probability
    return(n,corrects,target_history,distr_history,display)  


def get_coordinates(data,x_init,y_init,coordinates,grid_size):
    # Get the coordinates of a curve already drwan starting from (x_init,y_init)
    if len(coordinates) == 0:
        coordinates.append(x_init+y_init*grid_size)
    possible_new_point = [(x_init+1,y_init),(x_init-1,y_init),(x_init,y_init+1),(x_init,y_init-1)]
    data[x_init,y_init] = 0
    for i in range(len(possible_new_point)):
        if possible_new_point[i][0] >= 0 and possible_new_point[i][0] < grid_size and possible_new_point[i][1] >= 0 and possible_new_point[i][1] < grid_size:
            if data[possible_new_point[i][0],possible_new_point[i][1]] > 0:
                coordinates.append(possible_new_point[i][0]+possible_new_point[i][1]*grid_size)
                coordinates_temp = get_coordinates(data,possible_new_point[i][0],possible_new_point[i][1],coordinates,grid_size)
                coordinates = coordinates+coordinates_temp
    return(coordinates)

def get_extremity(data,pixels,grid_size):
    #Get the extremity of all the curves present on a numpy array
    for i in range(len(pixels[0])):
        x = pixels[0][i]
        y = pixels[1][i]
        possible_new_point = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        neighbours = 0
        for i in range(len(possible_new_point)):
            if possible_new_point[i][0] >= 0 and possible_new_point[i][0] < grid_size and possible_new_point[i][1] >= 0 and possible_new_point[i][1] < grid_size:
                if data[possible_new_point[i][0],possible_new_point[i][1]] > 0:
                    neighbours += 1
        if neighbours == 1:
            return(x,y)

def attention_dynamics(ar,CurveLength,grid_size,max_dur,corrects,object_1,object_2,cpu=False):
    # Get the dynamics of the spreading of attention for all the pixels of the objects if they are curve, only for the last one if they are object
    feat = 0  
    p = 1
    dur = len(ar[p])
    curves = [np.zeros(max(dur,max_dur)) for i in range(CurveLength)]
    if corrects[p-1] == 1:
        target_hist = object_1
        distr_hist = object_2
        for i in range(dur):
            if cpu:
                ar[p][i] = ar[p][i].cpu()
            if CurveLength > 1:
                curves[0][i] = ar[p][i][0,feat,target_hist[0] % grid_size, target_hist[0] //grid_size] - ar[p][i][0,feat,distr_hist[0]%grid_size,distr_hist[0]//grid_size]
                for l in range(1,CurveLength-1):
                    curves[l][i] = ar[p][i][0,feat,target_hist[l]%grid_size,target_hist[l]//grid_size] -  ar[p][i][0,feat,distr_hist[l]%grid_size,distr_hist[l]//grid_size]
            curves[CurveLength-1][i] = ar[p][i][0,feat,target_hist[-1]%grid_size,target_hist[-1]//grid_size] -  ar[p][i][0,feat,distr_hist[-1]%grid_size,distr_hist[-1]//grid_size]
            if dur < max_dur:
                if cpu:
                    ar[p][-1] = ar[p][-1].cpu()
                if CurveLength > 1:
                    curves[0][dur:] = ar[p][-1][0,feat,target_hist[0] % grid_size, target_hist[0] //grid_size] - ar[p][-1][0,feat,distr_hist[0]%grid_size,distr_hist[0]//grid_size]
                    for l in range(1,CurveLength-1):
                        curves[l][dur:] = ar[p][-1][0,feat,target_hist[l]%grid_size,target_hist[l]//grid_size] -  ar[p][-1][0,feat,distr_hist[l]%grid_size,distr_hist[l]//grid_size]
                curves[CurveLength-1][dur:] = ar[p][-1][0,feat,target_hist[-1]%grid_size,target_hist[-1]//grid_size] -  ar[p][-1][0,feat,distr_hist[-1]%grid_size,distr_hist[-1]//grid_size]
    for l in range(CurveLength):
        curves[l] = curves[l][~np.all(curves[l] == 0, axis=0)]
    return(curves)


def real_latency(curves,threshold,correct):
    # Get the actual latency for all the dynamics of attention spreading given by curves
    number_interpolation_points = 500
    latency_all = []
    for i in range(len(curves)):
        latency_interm = []
        max_dur =curves[i][0].shape[1] 
        CurveLength = len(curves[i])
        if correct[i]:
            for l in range(CurveLength):
                interpollation_function = scipy.interpolate.interp1d(np.arange(0,max_dur), curves[i][l], kind='linear',axis=-1)
                interpolated = interpollation_function(np.linspace(0, max_dur-1, num=number_interpolation_points))
                latency = np.where(interpolated > threshold*np.max(interpolated))[1][0]
                latency_interm.append(np.linspace(0, max_dur-1, num=number_interpolation_points)[latency])
            latency_all.append(latency_interm)
        else:
            latency_all.append(0)
    return(latency_all)

def distance_from_fixation_point(low_grid,middle_grid,high_grid, start,end,pixel_by_pixel=False):
    queue = collections.deque([start])
    width = low_grid.shape[0]
    height = low_grid.shape[1]
    low_grid[start[0],start[1]] = 1
    low_grid[end[0],end[1]] = 1
    middle_pixel_size = 3
    big_pixel_size = 9
    distance_grid = np.zeros_like(low_grid)
    distance_grid[start] = 1
    while queue:
        path = queue.popleft()
        x = path[0]
        y = path[1]
        possible_coordinates = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
        last_distance = distance_grid[x,y]
        for k in range(len(possible_coordinates)):
            x2 = possible_coordinates[k][0]
            y2 = possible_coordinates[k][1]
            if 0 <= x2 < width and 0 <= y2 < height and low_grid[x2][y2] > 0 and distance_grid[x2][y2] == 0 and last_distance>0:
                middle_coordinate = (x2 // middle_pixel_size, y2 // middle_pixel_size)
                big_coordinate = (x2 // big_pixel_size, y2 // big_pixel_size)
                if pixel_by_pixel:
                    distance_grid[x2,y2] = last_distance + 1
                    queue.append([x2,y2])
                else:
                    if high_grid[big_coordinate[0],big_coordinate[1]] > 0:
                        distance_grid_interm = distance_grid[big_coordinate[0]*big_pixel_size:big_coordinate[0]*big_pixel_size+big_pixel_size,big_coordinate[1]*big_pixel_size:big_coordinate[1]*big_pixel_size+big_pixel_size] 
                        low_grid_interm = low_grid[big_coordinate[0]*big_pixel_size:big_coordinate[0]*big_pixel_size+big_pixel_size,big_coordinate[1]*big_pixel_size:big_coordinate[1]*big_pixel_size+big_pixel_size]
                        distance_grid_interm[np.where(low_grid_interm==1)] = last_distance + 1
                        for i in range(big_pixel_size):
                            for j in range(big_pixel_size):
                                queue.append([big_coordinate[0]*big_pixel_size+i,big_coordinate[1]*big_pixel_size+j])
                    elif middle_grid[middle_coordinate[0],middle_coordinate[1]] > 0:
                        distance_grid_interm = distance_grid[middle_coordinate[0]*middle_pixel_size:middle_coordinate[0]*middle_pixel_size+middle_pixel_size,middle_coordinate[1]*middle_pixel_size:middle_coordinate[1]*middle_pixel_size+middle_pixel_size]
                        low_grid_interm = low_grid[middle_coordinate[0]*middle_pixel_size:middle_coordinate[0]*middle_pixel_size+middle_pixel_size,middle_coordinate[1]*middle_pixel_size:middle_coordinate[1]*middle_pixel_size+middle_pixel_size]
                        distance_grid_interm[np.where(low_grid_interm==1)] = last_distance + 1             
                        for i in range(middle_pixel_size):
                            for j in range(middle_pixel_size):
                                queue.append([middle_coordinate[0]*middle_pixel_size+i,middle_coordinate[1]*middle_pixel_size+j])
                    else:
                        distance_grid[x2,y2] = last_distance + 1
                        queue.append([x2,y2])
    return(distance_grid)
