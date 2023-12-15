'''
Created on 2023年10月17日

@author: sandy
'''
import pygame
import random
import math
import winsound

class Ball_Hit_Ball_Game:
    def __init__(self):
        # 设置常量
        self.RED = (213, 50, 80)
        self.BLUE = (50, 153, 213)
        self.GRAY = (128, 128, 128)
        self.GAME_WIDTH, self.GAME_HEIGHT = 400, 400
        self.TARGET_RADIUS, self.BULLET_RADIUS = 20, 20
        self.TARGET_VELOCITY = 10
        self.BULLET_VELOCITY = 20
        # 初始化变量
        self.game_over = False
        self.init_ball_location()
        # 初始化pygame
        self.screen = pygame.display.set_mode((self.GAME_WIDTH, self.GAME_HEIGHT))
        self.clock = pygame.time.Clock()
        pygame.init()
        pygame.key.set_repeat(10, 100)

    def init_ball_location(self):
        self.bullet_x = self.BULLET_RADIUS
        self.bullet_y = self.GAME_HEIGHT - self.BULLET_RADIUS
        self.target_x = random.randint(self.TARGET_RADIUS, self.GAME_WIDTH - self.TARGET_RADIUS)
        self.target_y = self.TARGET_RADIUS

    def collide(self):
        collide = False
        dist = math.sqrt((self.bullet_x - self.target_x)**2 + (self.bullet_y - self.target_y)**2)
        if dist < (self.BULLET_RADIUS + self.TARGET_RADIUS):
            collide = True
        return collide

    def play(self):
        # game loop
        while not self.game_over:

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.game_over = True
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                    self.bullet_x += self.BULLET_VELOCITY
                    if self.bullet_x > self.GAME_WIDTH - self.BULLET_RADIUS:
                        self.bullet_x = self.GAME_WIDTH - self.BULLET_RADIUS

            # 更新红球的位置
            self.target_y += self.TARGET_VELOCITY

            # 设置屏幕底色
            self.screen.fill(self.GRAY)

            # 更新屏幕
            target = pygame.draw.circle(self.screen, self.RED,
                                        (self.target_x, self.target_y), self.TARGET_RADIUS)
            bullet = pygame.draw.circle(self.screen, self.BLUE,
                                        (self.bullet_x, self.bullet_y), self.BULLET_RADIUS)

            # 检查蓝球是否碰到红球，必要时反馈回报并更新球的位置
            if self.collide():
                self.reward = 1
                winsound.Beep(3000, 100)
                self.init_ball_location()
            else:
                if self.target_y >= self.GAME_HEIGHT - self.TARGET_RADIUS:
                    if self.collide():
                        self.reward = 1
                        winsound.Beep(3000, 100)
                    else:
                        self.reward = -1
                    self.init_ball_location()

            pygame.display.flip()
            self.clock.tick(30)

        # 退出游戏
        pygame.quit()

if __name__ == "__main__":
    game = Ball_Hit_Ball_Game()
    game.play()
    
    
    
