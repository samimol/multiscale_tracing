# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 13:49:49 2023

@author: Sami
"""

import numpy as np
from scipy.special import binom
import random
from skimage.draw import polygon
import matplotlib.pyplot as plt

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


                        
