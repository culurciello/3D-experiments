from PIL import Image
import torch
import numpy as np
from pygame.locals import *
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *

# IMPORT OBJECT LOADER
from objloader import *


def solid_cube(pos=(0,0,0), size=(1,1,1)):
    cubeVertices = ((1,1,1),(1,1,-1),(1,-1,-1),(1,-1,1),(-1,1,1),(-1,-1,-1),(-1,-1,1),(-1,1,-1))
    cv = np.array(cubeVertices)
    cv = cv*size + pos
    cubeVertices = list(cv)
    cubeEdges = ((0,1),(0,3),(0,4),(1,2),(1,7),(2,5),(2,3),(3,6),(4,6),(4,7),(5,6),(5,7))
    cubeQuads = ((0,3,6,4),(2,5,6,3),(1,2,5,7),(1,0,4,7),(7,4,6,5),(2,3,0,1))

    glBegin(GL_QUADS)
    for cubeQuad in cubeQuads:
        for cubeVertex in cubeQuad:
            glVertex3fv(cubeVertices[cubeVertex])
    glEnd()


def init_pygame(width=500, height=500):
    pygame.init()
    pygame.display.set_mode((width, height), OPENGL | DOUBLEBUF)

    glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH) # most obj files expect to be smooth-shaded

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    gluPerspective(90.0, width/float(height), 1, 100.0)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_MODELVIEW)


def load_obj(filename):
    obj = OBJ(filename, swapyz=True, normalize=True)
    obj.generate()
    return obj

def set_viewpoint(r, zpos):
    glTranslate(0., 0., zpos)
    glRotate(r[1], 1, 0, 0)
    glRotate(r[0], 0, 1, 0)
    glRotate(r[2], 0, 0, 1)

def render_obj(obj, width=500, height=500, r=(0, 0, 0), zpos=-2):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    set_viewpoint(r, zpos)
    obj.render()
    buf = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    return buf

def render_cube(width=500, height=500, r=(0, 0, 0), zpos=-2):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    set_viewpoint(r, zpos)
    solid_cube()
    buf = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    return buf

def render_cubes(cubes_pos, cubes_size, width=500, height=500, r=(0, 0, 0), zpos=-2):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    set_viewpoint(r, zpos)
    for i,p in enumerate(cubes_pos):
        s = cubes_size[i]
        solid_cube(p,s)

    buf = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    return buf

def get_tensor_from_buffer(buf, width=500, height=500):
    tensor = torch.frombuffer(buf, dtype=torch.uint8)
    tensor = tensor.reshape(width, height, 3).float()/256.0
    return tensor


def save_render(filename="image.png", width=500, height=500):
    image_buffer = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.frombuffer("RGB", (width, height), image_buffer)
    image.save(filename)


def view_render():
    pygame.display.flip()
    pygame.time.wait(10)