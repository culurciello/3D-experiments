# test multi-cube render:
# https://stackabuse.com/advanced-opengl-in-python-with-pygame-and-pyopengl/

from renderer import init_pygame, save_render, render_cube, render_cubes

init_pygame()
# render_cube()
# cubes_pos = [(-1.5,1.5,0),(1.5,-1.5,0),(0,0,0)]
# cubes_size = [(1.3,1.3,1.3),(1,1,1),(0.7,0.7,0.7)]

cubes_pos = [(-1,1,0),(0,0,0),(1,1,0)]
cubes_size = [(0.5,0.5,0.5),(0.5,0.5,0.5),(0.5,0.5,0.5)]

buf = render_cubes(cubes_pos, cubes_size)
save_render(filename="test_cube.png")