import pygame
import random
import os
import math

class Ball_Hit_Ball_Game(object):
    
    def __init__(self):        
        # run pygame in headless mode        
        #os.environ["SDL_VIDEODRIVER"] = "dummy" 
        
        pygame.init()
        pygame.key.set_repeat(10, 100)
        
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

    def collide(self):
        collide = False
        dist = math.sqrt((self.bullet_x-self.target_x)**2+
                         (self.bullet_y-self.target_y)**2)
        if dist<(self.BULLET_RADIUS+self.TARGET_RADIUS):
            collide = True
        return collide             

    def reset(self):
        self.game_over = False
        self.reward = 0
        self.bullet_x = self.BULLET_RADIUS
        self.bullet_y = self.GAME_HEIGHT-self.BULLET_RADIUS    
        self.target_x = random.randint(self.TARGET_RADIUS, 
                             self.GAME_WIDTH-self.TARGET_RADIUS)        
        self.target_y = self.TARGET_RADIUS 
        self.screen = pygame.display.set_mode(
                (self.GAME_WIDTH, self.GAME_HEIGHT))
        self.clock = pygame.time.Clock()
    
    def step(self, action):
        # 更新蓝球的位置
        if action == 1: # move paddle right            
            self.bullet_x += self.BULLET_VELOCITY
            if self.bullet_x>self.GAME_WIDTH-self.BULLET_RADIUS:
                self.bullet_x=self.GAME_WIDTH-self.BULLET_RADIUS  
        else:             # dont move paddle
            pass        
        # 更新红球的位置
        self.target_y += self.TARGET_VELOCITY
        # 更新屏幕
        self.screen.fill(self.GRAY)           
        target = pygame.draw.circle(self.screen, self.RED,
                        (self.target_x, self.target_y), self.TARGET_RADIUS)
        bullet = pygame.draw.circle(self.screen, self.BLUE,
                  (self.bullet_x, self.bullet_y), self.BULLET_RADIUS) 
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
        image = pygame.surfarray.array3d(self.screen)  # 屏幕截图 (400,400,3)
        return image, self.reward, self.game_over    

if __name__ == "__main__":  
    import winsound
    game = Ball_Hit_Ball_Game()

    NUM_EPOCHS = 20
    for e in range(NUM_EPOCHS):
        game.reset()
        game_over = False
        while not game_over:
            action = random.randint(0, 2)
            input_tp1, reward, game_over = game.step(action)
            if reward==1:
                winsound.Beep(3000,100)  
    # 退出游戏    
    pygame.quit()