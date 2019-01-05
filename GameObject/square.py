import numpy as np
import  pygame
import time

from .entity import *

class Square(Entity):

    def __init__(self,x,y,w,h,color = (0,255,0)):


        super(Square,self).__init__(x,y)
        self.w = w
        self.h = h
        self.color = color

    def getRect(self):
        return pygame.Rect(self.x, self.y, self.w, self.h)


    # colllision test
    def collide(self,entity):

        return entity.x == self.x and self.y == entity.y

    def render(self,screen):
        pygame.draw.rect(screen, self.color, pygame.Rect(self.x, self.y, self.w, self.h))

