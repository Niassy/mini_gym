from .square import *

class MovingEntiy(Square):

    def __init__(self,x,y,w,h,color = (0,255,0)):

        super(MovingEntiy,self).__init__(x,y,w,h,color)

        self.speed = 32

    # direction is a tuple(x,y)
    def move(self,direction):

        moveX,moveY = direction[0] * self.speed,direction[1] * self.speed
        self.x ,self.y = self.x +moveX,self.y + moveY
