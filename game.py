import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
from scipy import interpolate

#rd.seed(42)

def rand_color():
    return (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))

def coords(angle_xy, angle_xz, radius, dtype=int):
    return np.array([radius*np.cos(angle_xy/180*np.pi)*np.cos(angle_xz/180*np.pi), -radius*np.sin(angle_xy/180*np.pi), radius*np.cos(angle_xy/180*np.pi)*np.sin(angle_xz/180*np.pi)], dtype=dtype)

def radius(x):
    return np.sqrt(np.sum(x*x))

def noize_3d_normal(mu, sigma):
    return np.array(np.random.normal(mu, sigma, 3), dtype=float)

class Ball:
    def __init__(self, time=float(0), start=np.array([0, 0, 0]), speed=float(10), angle_xy=10, angle_xz=20, g=np.array([0, -1, 0], dtype=float), beta=0.07, r=5, color=(100, 255, 0)):
        self.start = start
        self.start_time = time
        self.start_speed = coords(angle_xy, angle_xz, speed, float)
        self.g = g
        self.r = r
        self.color = color
        self.beta = beta
    def tic(self, time):
        #parabolical
        if self.beta==0:
            return self.start + (time-self.start_time)*self.start_speed + (time-self.start_time)**2/2*self.g
        #ballistic
        else:
            return self.start + 1/self.beta*(self.start_speed - self.g/self.beta)*(1-np.exp(-self.beta*(time-self.start_time))) + self.g/self.beta*(time-self.start_time)
def rand_point(xyz_min, xyz_max):
    return np.array([rd.randint(xyz_min[0], xyz_max[0]), rd.randint(xyz_min[1], xyz_max[1]), rd.randint(xyz_min[2], xyz_max[2])])

def cubic_movement_gen(xyz_min, xyz_max, max_time, max_target_speed, distance, start_point=None):
    if start_point is None:
        plan = [rand_point(xyz_min, xyz_max)]
    else:
        plan = [start_point]
    times = [0]
    if max_target_speed == 0:
        #static target
        x = lambda t: plan[0][0]
        y = lambda t: plan[0][1]
        z = lambda t: plan[0][2]
        return x, y, z
    else:
        i = 0
        distance3d = np.array([distance, distance, distance], dtype=int)
        while times[i]<1.5*max_time or len(times)<10:
            oldPoint = plan[i]
            new_xyz_min = np.maximum(xyz_min, oldPoint - distance3d)
            new_xyz_max = np.minimum(xyz_max, oldPoint + distance3d)
            new_point = rand_point(new_xyz_min, new_xyz_max)
            plan+=[new_point]
            time = times[i]
            times += [time + radius(new_point - oldPoint)/max_target_speed]
            i+=1
    plan = np.array(plan)
    #print(plan[:, 0])
    #print(times)
    x = interpolate.interp1d(times, plan[:, 0], kind='cubic', fill_value="extrapolate")
    y = interpolate.interp1d(times, plan[:, 1], kind='cubic', fill_value="extrapolate")
    z = interpolate.interp1d(times, plan[:, 2], kind='cubic', fill_value="extrapolate")
    return x, y, z
        
        
class GameViewer:
    def __init__(self, width=1000, height=600, target_xyz = None, max_time = 100, max_target_speed = 1, step_target_distance = 40, shoot_skip = 10):
        #windows is list with names of windows (len 0 or 1)
        self.windows = []
        self.name = 'MOUGAME'
        
        self.width, self.height = width, height
        self.shape = np.array([self.width, self.height], dtype=int)
        
        self.half_shape = self.shape//2
        self.plot_indent = np.min(self.half_shape//10)
        self.axis_indent = (self.plot_indent*np.array([-0.7, 0.5])).astype(int)
        self.plot_size = self.half_shape - 2*self.plot_indent
        
        #upper left angles of grids
        self.xy = (self.shape*[0, 0]).astype(int)+self.plot_indent
        self.yz = (self.shape*[0.5, 0]).astype(int)+self.plot_indent
        self.xz = (self.shape*[0, 0.5]).astype(int)+self.plot_indent
        
        #centers of grids
        self.xy0 = self.xy + (self.plot_size*[0.5, 0.5]).astype(int)
        self.yz0 = self.yz + (self.plot_size*[0.5, 0.5]).astype(int)
        self.xz0 = self.xz + (self.plot_size*[0.5, 0.5]).astype(int)
        
        self.xy0_relative = self.xy0 - self.xy
        self.yz0_relative = self.yz0 - self.yz
        self.xz0_relative = self.xz0 - self.xz
        
        self.text_indent = (self.shape*[0.5, 0.5]).astype(int)+self.plot_indent
        self.new_line_indent = np.array([0, 20], dtype=int)
        self.table_color = (200, 200, 0)
        self.text_color = (255, 255, 0)
        self.background_color = np.array([40, 60, 50], np.uint8)
        self.sphere_color = (40,150,40)
        self.grid_color = (40,150,40)
        #line thikness
        self.table_line_width = 1
        self.sphere_line_width = -1
        self.text_line_width = 1
        
        self.fontscale = 0.5
        
        #grid params
        self.grid_x = self.plot_size[0]//2//50
        self.grid_y = self.plot_size[1]//2//50
        self.grid_x_step = self.plot_size[0]//2//(self.grid_x+1)
        self.grid_y_step = self.plot_size[1]//2//(self.grid_y+1)
        
        #make blank background
        self.background = np.zeros((height,width,3), np.uint8) + self.background_color
        
        #divide into areas
        cv2.line(self.background, (self.shape*[0.5, 0]).astype(int), (self.shape*[0.5, 1]).astype(int), self.table_color, self.table_line_width)
        cv2.line(self.background, (self.shape*[0, 0.5]).astype(int), (self.shape*[1, 0.5]).astype(int), self.table_color, self.table_line_width)
        cv2.rectangle(self.background, self.xy, self.xy+self.plot_size,  self.table_color, self.table_line_width)
        cv2.rectangle(self.background, self.yz, self.yz+self.plot_size,  self.table_color, self.table_line_width)
        cv2.rectangle(self.background, self.xz, self.xz+self.plot_size,  self.table_color, self.table_line_width)
        
        #sign the axes
        cv2.putText(self.background, "y", self.xy + self.axis_indent, cv2.FONT_HERSHEY_SIMPLEX, 
                           self.fontscale, self.text_color, self.text_line_width, cv2.LINE_AA)
        cv2.putText(self.background, "y", self.yz + self.axis_indent, cv2.FONT_HERSHEY_SIMPLEX, 
                           self.fontscale, self.text_color, self.text_line_width, cv2.LINE_AA)
        cv2.putText(self.background, "z", self.xz + self.axis_indent, cv2.FONT_HERSHEY_SIMPLEX, 
                           self.fontscale, self.text_color, self.text_line_width, cv2.LINE_AA)
        cv2.putText(self.background, "x", self.xy + self.plot_size + self.axis_indent, cv2.FONT_HERSHEY_SIMPLEX, 
                           self.fontscale, self.text_color, self.text_line_width, cv2.LINE_AA)
        cv2.putText(self.background, "z", self.yz + self.plot_size + self.axis_indent, cv2.FONT_HERSHEY_SIMPLEX, 
                           self.fontscale, self.text_color, self.text_line_width, cv2.LINE_AA)
        cv2.putText(self.background, "x", self.xz + self.plot_size + self.axis_indent, cv2.FONT_HERSHEY_SIMPLEX, 
                           self.fontscale, self.text_color, self.text_line_width, cv2.LINE_AA)
        #plot grid
        for i in range(-self.grid_y, self.grid_y+1):
            cv2.line(self.background, [self.xy[0], self.xy0[1] +i*self.grid_x_step], [self.xy[0]+self.plot_size[0], self.xy0[1] +i*self.grid_x_step], self.grid_color, self.table_line_width)
            cv2.line(self.background, [self.yz[0], self.yz0[1] +i*self.grid_x_step], [self.yz[0]+self.plot_size[0], self.yz0[1] +i*self.grid_x_step], self.grid_color, self.table_line_width)
            cv2.line(self.background, [self.xz[0], self.xz0[1] +i*self.grid_x_step], [self.xz[0]+self.plot_size[0], self.xz0[1] +i*self.grid_x_step], self.grid_color, self.table_line_width)
        for i in range(-self.grid_x-1, self.grid_x+2):
            cv2.line(self.background, [self.xy0[0]+i*self.grid_y_step, self.xy[1]], [self.xy0[0]+i*self.grid_y_step, self.xy[1]+self.plot_size[1]], self.grid_color, self.table_line_width)
            cv2.line(self.background, [self.yz0[0]+i*self.grid_y_step, self.yz[1]], [self.yz0[0]+i*self.grid_y_step, self.yz[1]+self.plot_size[1]], self.grid_color, self.table_line_width)
            cv2.line(self.background, [self.xz0[0]+i*self.grid_y_step, self.xz[1]], [self.xz0[0]+i*self.grid_y_step, self.xz[1]+self.plot_size[1]], self.grid_color, self.table_line_width)
            
        
        self.balls = []
        self.emitter_r = np.min(self.plot_size)//10
        self.emitter_line_width = 4
        self.emitter_color = (0, 0, 255)
        
        self.target_color = (0, 0, 255)
        self.target_r = 10
        xyz_min = np.array([-self.plot_size[0]//2+self.target_r, -self.plot_size[1]//2+self.target_r, -self.plot_size[1]//2+self.target_r], dtype=int)
        xyz_max = -xyz_min
        self.target_x, self.target_y, self.target_z = cubic_movement_gen(xyz_min, xyz_max, max_time, max_target_speed, step_target_distance, target_xyz) 
        self.target_xyz = np.array([self.target_x(0), self.target_y(0), self.target_z(0)], dtype=int)
        
        ## Model
        self.angle_xz = float(0)
        self.angle_xy = float(45)
        self.time = float(0)
        self.dt = float(0.1)
        self.step = 0
        self.speed = float(20)
        self.g = np.array([0, 1, 0], dtype=float)
        self.beta = 0.07
        self.ball_r = 5
        
        self.shoot_skip = shoot_skip
        
        self.score = 0
        ## Noize
        self.mu = 0
        self.sigma = self.target_r/2
        
    def tic(self, angle_xy, angle_xz):
        self.angle_xy, self.angle_xz = angle_xy, angle_xz
        self.step += 1
        self.time = self.dt*self.step
        self.target_xyz = np.array([self.target_x(self.time), self.target_y(self.time), self.target_z(self.time)], dtype=int)
        if self.step % self.shoot_skip == 0:
            self.balls.append(Ball(self.time, coords(self.angle_xy, self.angle_xz, self.emitter_r, float), 
                                   self.speed, self.angle_xy, self.angle_xz, self.g, self.beta, self.ball_r, rand_color()))#color=(100, 255, 0)))
        to_del = []
        for i, b in enumerate(self.balls):
            if radius(b.tic(self.time) - self.target_xyz)<=self.target_r+b.r:
                del self.balls[i]
                self.score += self.shoot_skip/10
        return self.target_xyz + noize_3d_normal(self.mu, self.sigma), self.score, self.angle_xy, self.angle_xz, self.speed, self.time
        
    
    def view(self, text = ['text', 'text']):
        text = ['SCORE: %.1f'%self.score, 'speed_0: %.1f'%self.speed, 'angle_xy: %.1f'%(self.angle_xy%360), 'angle_xz: %.1f'%(self.angle_xz%360)]
        im = self.background.copy()
        im_xy = im[self.xy[1]:self.xy[1]+self.plot_size[1], self.xy[0]:self.xy[0]+self.plot_size[0]]
        im_yz = im[self.yz[1]:self.yz[1]+self.plot_size[1], self.yz[0]:self.yz[0]+self.plot_size[0]]
        im_xz = im[self.xz[1]:self.xz[1]+self.plot_size[1], self.xz[0]:self.xz[0]+self.plot_size[0]]
        #balls
        for b in self.balls:
            xyz = b.tic(self.time).astype(int)
            xy = xyz[[0, 1]]
            yz = xyz[[2, 1]]
            xz = xyz[[0, 2]]
            #print(xyz, b.r)
            cv2.circle(im_xy, self.xy0_relative+xy*[1, 1], b.r, b.color, self.sphere_line_width)
            cv2.circle(im_yz, self.yz0_relative+yz*[1, 1], b.r, b.color, self.sphere_line_width)
            cv2.circle(im_xz, self.xz0_relative+xz*[1, 1], b.r, b.color, self.sphere_line_width)
        xy = self.target_xyz[[0, 1]].astype(int)
        yz = self.target_xyz[[2, 1]].astype(int)
        xz = self.target_xyz[[0, 2]].astype(int)
        #print(xyz, self.target_r)
        cv2.circle(im_xy, self.xy0_relative+xy*[1, 1], self.target_r, self.target_color, self.sphere_line_width)
        cv2.circle(im_yz, self.yz0_relative+yz*[1, 1], self.target_r, self.target_color, self.sphere_line_width)
        cv2.circle(im_xz, self.xz0_relative+xz*[1, 1], self.target_r, self.target_color, self.sphere_line_width)
        
        #gun
        cv2.line(im, self.xy0, self.xy0+coords(self.angle_xy, self.angle_xz, self.emitter_r)[[0, 1]], self.emitter_color, self.emitter_line_width)
        cv2.line(im, self.yz0, self.yz0+coords(self.angle_xy, self.angle_xz, self.emitter_r)[[2, 1]], self.emitter_color, self.emitter_line_width)
        cv2.line(im, self.xz0, self.xz0+coords(self.angle_xy, self.angle_xz, self.emitter_r)[[0, 2]], self.emitter_color, self.emitter_line_width)
        #text
        for i, text_line in enumerate(text):
            cv2.putText(im, text_line, self.text_indent + self.new_line_indent*(i), cv2.FONT_HERSHEY_SIMPLEX, 
                           self.fontscale, self.text_color, self.text_line_width, cv2.LINE_AA)
            
        if self.name not in self.windows:
            self.windows.append(self.name)
            cv2.namedWindow(str(self.name), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(self.name), im.shape[1], im.shape[0])
        cv2.imshow(str(self.name), im)
        cv2.waitKey(1)
        #return im
        
    
