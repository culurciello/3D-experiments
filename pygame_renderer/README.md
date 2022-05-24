# OBJFileLoader

A library from pygame wiki https://www.pygame.org/wiki/OBJFileLoader

from https://github.com/yarolig/OBJFileLoader


## How to use it with pygame and old OpenGL

Right now this library requires pygame, PyOpenGL and OpenGL compatibility profile (or OpenGL 2.0).

Copy OBJFileLoader into your project.

Import it:

    from OBJFileLoader import OBJ

After you initialise pygame and PyOpenGL you can load objects.

    box = OBJ('path/to/box.obj')
    
Material and texture files should be inside 'path/to' directory.
    
To draw the object position it where you want and call render().

    glPushMatrix()
    glTranslatef(box_x, box_y, box_z)
    box.render()
    glPopMatrix()
   
If you need to change behavior of texture loading or use VBO instead of lists create a subclass and override loadTexture() or generate().

## Viewer

To view an OBJ-file run:

    python view.py assets/Sofa/Sofa.obj


## Swap Texture:

this can be done by simply changing the image file / replace with new texture
or by updating the .mtl  file with the next texture, for example:


```
# Blender MTL File: 'Sofa.blend'
# Material Count: 3

newmtl Material
map_Kd leather1.jpeg

Ka 1.000 1.000 1.000
Kd 1.000 1.000 1.000
Ks 0.000 0.000 0.000
Ns 10.0

newmtl Material.001
map_Kd leather1.jpeg

Ka 1.000 1.000 1.000
Kd 1.000 1.000 1.000
Ks 0.000 0.000 0.000
Ns 10.0

newmtl Material.002
map_Kd leather1.jpeg

Ka 1.000 1.000 1.000
Kd 1.000 1.000 1.000
Ks 0.000 0.000 0.000
Ns 10.0

```
