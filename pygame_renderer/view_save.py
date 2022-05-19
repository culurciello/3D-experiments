#!/usr/bin/env python
# Basic OBJ file viewer. needs objloader from:
#  http://www.pygame.org/wiki/OBJFileLoader

# LMB + move: rotate
# RMB + move: pan
# Scroll wheel: zoom in/out

import sys
import pygame
# import numpy as np
from PIL import Image
from pygame.locals import *
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *

# IMPORT OBJECT LOADER
from objloader import *

pygame.init()

viewport = (500,500)
hx = viewport[0]/2
hy = viewport[1]/2
srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)

glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
glEnable(GL_LIGHT0)
glEnable(GL_LIGHTING)
glEnable(GL_COLOR_MATERIAL)
glEnable(GL_DEPTH_TEST)
glShadeModel(GL_SMOOTH) # most obj files expect to be smooth-shaded

# LOAD OBJECT AFTER PYGAME INIT
obj = OBJ(sys.argv[1], swapyz=True)
obj.generate()

# clock = pygame.time.Clock()

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
width, height = viewport
gluPerspective(90.0, width/float(height), 1, 100.0)
glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_MODELVIEW)

rx, ry, rz = (0, 0, 0) # rotation of view
zpos = -80

glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
glLoadIdentity()

# RENDER OBJECT
glTranslate(0., 0., zpos)
glRotate(ry, 1, 0, 0)
glRotate(rx, 0, 1, 0)
glRotate(rz, 0, 0, 1)
obj.render()

image_buffer = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
# image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(width, height, 3)
image = Image.frombuffer("RGB", (width, height), image_buffer)
# image = image.transpose( Image.FLIP_TOP_BOTTOM)
image.save("image.png")

# pygame.display.flip()
