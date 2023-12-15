
from __future__ import division, print_function
import collections
import numpy as np
import pygame
import random
import os
import math

class MyWrappedGame(object):
    
    def __init__(self):

        
        # set constants
        self.RED = (213, 50, 80)
        self.BLUE = (50, 153, 213)
        self.GRAY = (128,128,128)    
        self.GAME_WIDTH = 400
        self.GAME_HEIGHT = 400
        self.TARGET_RADIUS = 20
        self.BULLET_RADIUS = 20
        self.TARGET_VELOCITY = 10
        self.BULLET_VELOCITY = 20       
        
        self.game_over = False 
        self.init_ball_location()
        
        self.screen = pygame.display.set_mode(
                (self.GAME_WIDTH, self.GAME_HEIGHT))
        self.clock = pygame.time.Clock()
        
        pygame.init()
        pygame.key.set_repeat(10,100)

    def collide(self):
        collide = False
        dist = math.sqrt((self.bullet_x-self.target_x)**2+
                         (self.bullet_y-self.target_y)**2)
        if dist<(self.BULLET_RADIUS+self.TARGET_RADIUS):
            collide = True
        return collide             

    def reset(self):

        self.game_over = False        
        # initialize positions
        self.bullet_x = self.BULLET_RADIUS
        self.game_score = 0
        self.reward = 0
        self.target_x = random.randint(self.TARGET_RADIUS, 
                                     self.GAME_WIDTH-self.TARGET_RADIUS)
        self.bullet_y = self.GAME_HEIGHT-self.BULLET_RADIUS         
        self.target_y = self.TARGET_RADIUS         
        self.num_tries = 0
        # set up display, clock, etc
        self.screen = pygame.display.set_mode(
                (self.GAME_WIDTH, self.GAME_HEIGHT))
        self.clock = pygame.time.Clock()
    
    def step(self, action):
        pygame.event.pump()

        if action == 1: # move paddle right            
            self.bullet_x += self.BULLET_VELOCITY
            if self.bullet_x>self.GAME_WIDTH-self.BULLET_RADIUS:
                self.bullet_x=self.GAME_WIDTH-self.BULLET_RADIUS  
        else:             # dont move paddle
            pass

        self.screen.fill(self.GRAY)
           
        
        # 更新球的位置
        self.target_y += self.TARGET_VELOCITY
        target = pygame.draw.circle(self.screen, self.RED,
                        (self.target_x, self.target_y), self.TARGET_RADIUS)
        bullet = pygame.draw.circle(self.screen, self.BLUE,
                  (self.bullet_x, self.bullet_y), self.BULLET_RADIUS)   
        
        self.reward = 0
        # 检查是否相碰，并更新回报
        
        if self.collide():        
            self.reward = 1
            self.game_over = True
        else:
            if self.target_y >= self.GAME_HEIGHT - self.TARGET_RADIUS:
                if self.collide():        
                    self.reward = 1
                    self.game_over = True
                else:
                    self.reward = -1    
                    self.game_over = True    
                
        pygame.display.flip()
            
        self.clock.tick(30)

        # 屏幕截图      
        image = pygame.surfarray.array3d(self.screen)  #(400,400,3)
        
        return image, self.reward, self.game_over
    

if __name__ == "__main__":   
    game = MyWrappedGame()

    NUM_EPOCHS = 10
    for e in range(NUM_EPOCHS):
        print("Epoch: {:d}".format(e))
        game.reset()
        input_t = game.get_frames()
        game_over = False
        while not game_over:
            action = np.random.randint(0, 3, size=1)[0]
            input_tp1, reward, game_over = game.step(action)
            print(action, reward, game_over)
        
    