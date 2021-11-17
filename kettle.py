import numpy as np
import taichi as ti
import matplotlib.cm as cm

ti.init(arch=ti.gpu)


kettle = ti.imread("kettle.png")
size_x, size_y = kettle.shape[:2] # size of the domain
alpha = np.asarray(kettle[:, :, 0], np.float32)


delta_x = 0.25  # scaling (mm/px)
delta_t = (delta_x ** 2)/(4 * alpha.max()) # numerical stability constraint

gamma = ti.field(dtype=float, shape=(size_x, size_y))
gamma.from_numpy((alpha * delta_t) / (delta_x ** 2)) # pre-computed constants


bc = np.zeros((size_x, size_y), dtype=np.float32)
bc[:, 0] = 10.0    # bottom

t = ti.field(dtype=float, shape=(size_x, size_y))
t_next = ti.field(dtype=float, shape=t.shape)
t.from_numpy(bc)
t_next.from_numpy(bc)

@ti.kernel
def diffusion(t: ti.template(), t_next: ti.template()):
    for i, j in t:
        if 0 < i < size_x - 1 and 0 < j < size_y - 1:
            t_next[i, j] = t[i, j] + gamma[i, j] * (
                t[i+1, j] + t[i-1, j] + t[i, j+1] + t[i, j-1] - 4*t[i, j])


gui = ti.GUI("Heatsim", res=(size_x, size_y))
cmap = cm.get_cmap("inferno")
while not gui.get_event(ti.GUI.ESCAPE):
    diffusion(t, t_next)

    if gui.is_pressed(ti.GUI.LMB):
        mouse_x, mouse_y = (gui.get_cursor_pos())
        t_next[int(mouse_x * size_x), int(mouse_y * size_y)] += 10.0

    t, t_next = t_next, t
    gui.set_image(cmap(t.to_numpy()))
    gui.show()
