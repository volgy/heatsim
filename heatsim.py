import numpy as np
import taichi as ti

ti.init(arch=ti.gpu)


size_x, size_y = 1024, 768 # size of the domain
delta_x = 1  # scaling
alpha = 2  # material diffusivity
delta_t = (delta_x ** 2)/(4 * alpha) # numerical stability constraint

# Initial condition
t0 = 0.0

# Boundary conditions (fixed temperature)
t_top = 0.0
t_left = 0.0
t_bottom = 1.0
t_right = 0.0

gamma = (alpha * delta_t) / (delta_x ** 2)


initial = np.full((size_x, size_y), t0, dtype=np.float32)
initial[: , -1] = t_top
initial[:, 0] = t_bottom
initial[0, :] = t_left
initial[-1, :] = t_right

t = ti.field(dtype=float, shape=(size_x, size_y))
t_next = ti.field(dtype=float, shape=t.shape)
t.from_numpy(initial)
t_next.from_numpy(initial)



@ti.kernel
def diffusion(t: ti.template(), t_next: ti.template()):
    for i, j in t:
        if 0 < i < size_x - 1 and 0 < j < size_y - 1:
            t_next[i, j] = t[i, j] + gamma * (
                t[i+1, j] + t[i-1, j] + t[i, j+1] + t[i, j-1] - 4*t[i, j])


gui = ti.GUI("Heatsim", res=(size_x, size_y))
while not gui.get_event(ti.GUI.ESCAPE):
    diffusion(t, t_next)

    if gui.is_pressed(ti.GUI.LMB):
        mouse_x, mouse_y = (gui.get_cursor_pos())
        t_next[int(mouse_x * size_x), int(mouse_y * size_y)] += 10.0

    t, t_next = t_next, t
    gui.set_image(t)
    gui.show()
