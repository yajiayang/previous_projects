# Learn python class development
import math
class Circle(object):
    'An advanced circle analytic toolkit'
    
    # flyweight design pattern suppresses
    # the instance dictionary do it last

    __slot__ = ['diameter']    
    
    version = '0.7'    
    
    def __init__(self,radius):
        self.radius = radius
        
    @property # convert dotted access to method calls
    def radius(self):
        'radius of a circle'
        return self.diameter / 2.0
    
    @staticmethod #attach functions to classes
    def angle_to_grade(angle):
        'Convert angle in degree to a percentage grade'
        return math.tan(math.radians(angle)) * 100.0
        
    def area(self):
        'perform quadrature on a shape of uniform radium'
        p = self.perimeter()
        r = p / math.pi / 2.0 
        return math.pi * r ** 2.0
        
    def perimeter(self):
        return 2.0 * math.pi * self.radius ** 2.0
        
    __perimeter=perimeter
        
    @classmethod #alternative constructor
    def from_bbd(cls, bbd):
        'Construct a circle from a bounding box diagonal'
        radius = bbd / 2.0 / math.sqrt(2.0)
        return cls(radius)

#subclass        
class Tire(Circle):
    'Tires are circles with a corrected perimeter'
    
    def perimeter(self):
        'circumference corrected for the rubber'
        return Circle.perimeter(self) * 1.25


print 'Circuituous version', Circle.version
c=Circle(10)
print 'A circle of radius', c.radius
print 'has an area of', c.area()

from random import random, seed
seed(8675309)
print 'Using Circuituous(tm) version', Circle.version
n=10
circles = [Circle(random()) for i in xrange(n)]
print 'the average area of', n, 'random circles'
avg = sum(c.area() for c in circles)/n
print 'is %.lf' % avg

Tire.from_bbd(45).perimeter()
t = Tire(22)
t.radius
t.area()
t.perimeter()
