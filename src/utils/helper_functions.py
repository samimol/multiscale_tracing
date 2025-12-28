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

def make_curves(curve, mask_original, curve_length, grid_size=3, direction=None):
    """Recursively generate a random curve on a grid.
    
    Args:
        curve (list): Current curve being built (empty to start).
        mask_original (np.ndarray): Mask of occupied pixels.
        curve_length (int): Desired length of the curve.
        grid_size (int): Size of the grid.
        direction (int, optional): Preferred direction (0=vertical, 1=horizontal).
        
    Returns:
        tuple: (curve, mask) - list of pixel indices and updated mask.
    """
    # Create a copy to avoid modifying the original mask
    mask = mask_original.copy()
    
    if len(curve) == 0:
        # Base case: initialize curve with a random starting pixel
        # Find all unoccupied pixels
        x, y = np.where(mask == 0)
        
        # Choose a random starting point
        ind = np.random.randint(len(x))
        # Convert 2D coordinates to 1D index: index = x + y * grid_size
        first_elem = x[ind] + y[ind] * grid_size
        
        # Mark this pixel as occupied and add to curve
        mask[first_elem % grid_size, first_elem // grid_size] = 1
        curve.append(first_elem)
        # Recursive call to continue building the curve
        return make_curves(curve, mask,curve_length,grid_size=grid_size,direction=direction)
    else:
        # Recursive case: extend the curve from its current endpoint
        # Get coordinates of the last pixel in the curve
        x_end = curve[-1] % grid_size
        y_end = curve[-1] // grid_size

        # Find valid horizontal neighbors (left and right)
        consecutives_x = [x_end - 1, x_end + 1]
        consecutives_x = [i for i in consecutives_x if i >= 0 and i < grid_size]
        
        # Find valid vertical neighbors (up and down)
        consecutives_y = [y_end - 1, y_end + 1]
        consecutives_y = [i for i in consecutives_y if i >= 0 and i < grid_size]

        # Combine all neighbors (4-connectivity: left, right, up, down)
        neighbors = [(consecutives_x[i],y_end) for i in range(len(consecutives_x))]+[(x_end,consecutives_y[i]) for i in range(len(consecutives_y))]
        # Track direction of each neighbor: 0=vertical movement, 1=horizontal movement
        directions = [0 for i in range(len(consecutives_x))] + [1 for i in range(len(consecutives_y))]
        
        
        # Continue growing the curve until desired length is reached
        while len(curve) < curve_length:  
            possible_next_value = []
            possible_next_value_directions = []
            
            # Find all unoccupied neighbors that can be added to the curve
            for i in range(len(neighbors)):
                if mask[neighbors[i]] == 0:
                    possible_next_value.append(neighbors[i])
                    possible_next_value_directions.append(directions[i])
            
            # Select next pixel based on direction constraint (if any)
            if direction is None:
                # No direction constraint: choose randomly from all available neighbors
                next_value = random.choice(possible_next_value)
            else:
                # Direction constraint: only choose neighbors in the specified direction
                # This creates more aligned curves (all vertical or all horizontal)
                possible_next_value = np.array(possible_next_value)
                possible_next_value_directions = np.array(possible_next_value_directions)
                next_value = random.choice(possible_next_value[possible_next_value_directions == direction])
            
            # Add the selected pixel to the curve
            curve.append(next_value[0] + next_value[1] * grid_size)
            
            # Mark all possible neighbors as occupied to prevent branching
            for i in range(len(possible_next_value)):
                mask[possible_next_value[i][0], possible_next_value[i][1]] = 1
            
            # Recursive call to continue growing
            return make_curves(curve, mask,curve_length,grid_size=grid_size,direction=direction)
        else:
            # Termination: curve has reached desired length
            # Mark all neighbors as occupied to prevent other curves from touching
            for i in range(len(neighbors)):
                mask[neighbors[i]] = 1
            return curve, mask
        



bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    """Generate a Bezier curve from control points.
    
    Args:
        points (np.ndarray): Control points for the Bezier curve.
        num (int): Number of points to sample along the curve.
        
    Returns:
        np.ndarray: Curve points of shape (num, 2).
    """
    N = len(points)  # Number of control points
    # Parameter t ranges from 0 (start) to 1 (end)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    
    # Bezier curve formula: sum of control points weighted by Bernstein polynomials
    # B(t) = sum_{i=0}^{N-1} P_i * bernstein(N-1, i, t)
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    """Bezier curve segment between two points with specified angles."""
    
    def __init__(self, p1, p2, angle1, angle2, **kw):
        """Initialize segment.
        
        Args:
            p1 (np.ndarray): Start point.
            p2 (np.ndarray): End point.
            angle1 (float): Angle at start point.
            angle2 (float): Angle at end point.
            **kw: Additional parameters (numpoints, r).
        """
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
        # Calculate control points for cubic Bezier curve
        # Control point 1: offset from start point in direction of angle1
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        # Control point 2: offset from end point in opposite direction of angle2
        # (angle2 + pi reverses the direction)
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        # Generate the smooth curve through these 4 control points
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
    # Compute smoothness parameter from edginess
    p = np.arctan(edgy)/np.pi+.5
    
    # Sort points in counter-clockwise order to create closed shape
    a = ccw_sort(a)
    # Close the curve by appending the first point at the end
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    
    # Calculate angles between consecutive points
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    
    # Normalize angles to [0, 2π]
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    
    # Smooth angles by blending with neighboring angles
    ang1 = ang
    ang2 = np.roll(ang,1)  # Shift angles by one position
    # Interpolate between consecutive angles, handling wraparound at 2π
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    
    # Append angles to point array for Bezier curve generation
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
    
def make_blob(grid_size,center,scale):
    # Generate a blob-like shape using Bezier curves
    rad = 0.9999  # High radius for smooth curves
    edgy = 0.9999  # High edginess for more irregular shapes
    n = 7  # Number of control points
    
    # Random center position for the blob
    c = [np.random.randint(center),np.random.randint(center)]
    
    # Generate random control points around the center
    a = get_random_points(n=n, scale=scale) + c
    
    # Create smooth closed curve through control points
    x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)
    
    # Fill the interior of the curve to create a solid blob
    x_blob,y_blob = polygon(x, y, (grid_size,grid_size))
    return x_blob,y_blob


def test_network(t,CurveLength,grid_size,TrialNumber,n,device,save_activities,only_blue,verbose=False):
    # Test network performance without learning (pure exploitation)
    # Save original exploration rate and set to pure exploitation
    prev_exploitation_probability = n.exploitation_probability
    n.exploitation_probability = 1  # Always choose best action (no exploration)
    
    # Configure activity saving
    n.save_activities = save_activities
    
    # Initialize tracking variables
    corrects = []  # Success/failure for each trial
    target_history = []  # Target curves for each trial
    distr_history = []  # Distractor curves for each trial
    display = []  # Visual displays for each trial
    
    # Set task parameters
    t.curve_length = CurveLength
    t.grid_size = grid_size
    t.only_blue = only_blue
    action = 0
    
    # Initialize activity storage if needed
    if save_activities:
        n.saved_activities = []
        for layer in range(n.num_scales+2):
            n.saved_activities.append([[]])
    
    # Run test trials
    for p in range(TrialNumber):
      trial_running = True
      display.append([])
      # Initialize trial
      new_input, reward, trialEnd= t.step(action)
      
      # Run trial until completion
      while trial_running:
        # Network selects action based on current input
        action = n.step(new_input,reward,trialEnd,device)
        # Task provides new input and reward
        new_input, reward, trialEnd = t.step(action)
        display[p].append(new_input)
        
        if trialEnd:
          trial_running = False
          # Record success (reward > 0) or failure (reward = 0)
          if reward == 0:
            corrects.append(0)
          else:
            corrects.append(1)
      
      # Store trial information
      target_history.append(t.target_curve)
      distr_history.append(t.distractor_curve)
    
    if verbose:
        print(np.mean(corrects))
    
    # Restore original exploration rate
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

def attention_dynamics(ar,CurveLength,grid_size,max_dur,corrects,object_1,object_2,to_cpu=False):
    # Get the dynamics of the spreading of attention for all the pixels of the objects if they are curve, only for the last one if they are object
    feat = 0  
    p = 1
    dur = len(ar[p])
    curves = [np.zeros(max(dur,max_dur)) for i in range(CurveLength)]
    if corrects[p-1] == 1:
        target_hist = object_1
        distr_hist = object_2
        for i in range(dur):
            if to_cpu:
                ar[p][i] = ar[p][i].cpu()
            if CurveLength > 1:
                curves[0][i] = ar[p][i][0,feat,target_hist[0] % grid_size, target_hist[0] //grid_size] - ar[p][i][0,feat,distr_hist[0]%grid_size,distr_hist[0]//grid_size]
                for l in range(1,CurveLength-1):
                    curves[l][i] = ar[p][i][0,feat,target_hist[l]%grid_size,target_hist[l]//grid_size] -  ar[p][i][0,feat,distr_hist[l]%grid_size,distr_hist[l]//grid_size]
            curves[CurveLength-1][i] = ar[p][i][0,feat,target_hist[-1]%grid_size,target_hist[-1]//grid_size] -  ar[p][i][0,feat,distr_hist[-1]%grid_size,distr_hist[-1]//grid_size]
            if dur < max_dur:
                if to_cpu:
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
                try:
                    latency = np.where(interpolated > threshold*np.max(interpolated))[1][0]
                    latency_interm.append(np.linspace(0, max_dur-1, num=number_interpolation_points)[latency])
                except:
                    latency_interm.append(0)
            latency_all.append(latency_interm)
        else:
            latency_all.append([0])
    return(latency_all)

def distance_from_fixation_point(low_grid,middle_grid,high_grid, start,end,pixel_by_pixel=False):
    """Compute distance from fixation point using multi-scale breadth-first search.
    
    This function simulates how attention spreads from a starting point,
    using coarse-to-fine processing: it jumps quickly through regions with
    high-scale activity, then processes medium-scale, then fine-scale.
    """
    # Initialize BFS queue with starting position
    queue = collections.deque([start])
    width = low_grid.shape[0]
    height = low_grid.shape[1]
    
    # Mark start and end points as valid
    low_grid[start[0],start[1]] = 1
    low_grid[end[0],end[1]] = 1
    
    # Define scale sizes (3x3 and 9x9 pooling)
    middle_pixel_size = 3
    big_pixel_size = 9
    
    # Distance grid tracks how many steps from fixation point
    distance_grid = np.zeros_like(low_grid)
    distance_grid[start] = 1  # Start has distance 1
    
    # Breadth-first search to propagate distances
    while queue:
        # Get next position from queue
        path = queue.popleft()
        x = path[0]
        y = path[1]
        
        # Check 4-connected neighbors
        possible_coordinates = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
        last_distance = distance_grid[x,y]
        
        for k in range(len(possible_coordinates)):
            x2 = possible_coordinates[k][0]
            y2 = possible_coordinates[k][1]
            
            # Check if neighbor is valid and unvisited
            if 0 <= x2 < width and 0 <= y2 < height and low_grid[x2][y2] > 0 and distance_grid[x2][y2] == 0 and last_distance>0:
                # Map to coarser scales
                middle_coordinate = (x2 // middle_pixel_size, y2 // middle_pixel_size)
                big_coordinate = (x2 // big_pixel_size, y2 // big_pixel_size)
                
                if pixel_by_pixel:
                    # Fine-scale processing: spread one pixel at a time
                    distance_grid[x2,y2] = last_distance + 1
                    queue.append([x2,y2])
                else:
                    # Multi-scale processing: jump through active regions
                    if high_grid[big_coordinate[0],big_coordinate[1]] > 0:
                        # Coarse scale is active: process entire 9x9 block at once
                        # This simulates rapid attention spread through salient regions
                        distance_grid_interm = distance_grid[big_coordinate[0]*big_pixel_size:big_coordinate[0]*big_pixel_size+big_pixel_size,big_coordinate[1]*big_pixel_size:big_coordinate[1]*big_pixel_size+big_pixel_size] 
                        low_grid_interm = low_grid[big_coordinate[0]*big_pixel_size:big_coordinate[0]*big_pixel_size+big_pixel_size,big_coordinate[1]*big_pixel_size:big_coordinate[1]*big_pixel_size+big_pixel_size]
                        distance_grid_interm[np.where(low_grid_interm==1)] = last_distance + 1
                        # Add all pixels in block to queue
                        for i in range(big_pixel_size):
                            for j in range(big_pixel_size):
                                queue.append([big_coordinate[0]*big_pixel_size+i,big_coordinate[1]*big_pixel_size+j])
                    
                    elif middle_grid[middle_coordinate[0],middle_coordinate[1]] > 0:
                        # Medium scale is active: process 3x3 block
                        distance_grid_interm = distance_grid[middle_coordinate[0]*middle_pixel_size:middle_coordinate[0]*middle_pixel_size+middle_pixel_size,middle_coordinate[1]*middle_pixel_size:middle_coordinate[1]*middle_pixel_size+middle_pixel_size]
                        low_grid_interm = low_grid[middle_coordinate[0]*middle_pixel_size:middle_coordinate[0]*middle_pixel_size+middle_pixel_size,middle_coordinate[1]*middle_pixel_size:middle_coordinate[1]*middle_pixel_size+middle_pixel_size]
                        distance_grid_interm[np.where(low_grid_interm==1)] = last_distance + 1
                        # Add all pixels in block to queue
                        for i in range(middle_pixel_size):
                            for j in range(middle_pixel_size):
                                queue.append([middle_coordinate[0]*middle_pixel_size+i,middle_coordinate[1]*middle_pixel_size+j])
                    
                    else:
                        # No coarse activity: process single pixel (slow spread)
                        distance_grid[x2,y2] = last_distance + 1
                        queue.append([x2,y2])
    
    return(distance_grid)

