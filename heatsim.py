#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Heat equation in 2D using the explicit method."""

# Copyright (c) 2021 Peter Volgyesi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import taichi as ti
import utils

ti.init(arch=ti.gpu)
# ti.aot.start_recording('heatsim.yml')
# ti.init(arch=ti.cc)

# k: thermal conductivity, c: heat capacity (per volume), s: heat source
materials = {
    0: {'k': 0.01, 'c': 0.1, 's': 0.0},     # air
    128: {'k': 0.7, 'c': 0.6, 's': 1.0},    # heater
    192: {'k': 0.9, 'c': 0.8 , 's': 0.0},   # kettle
}

c, k, s, outline = utils.load_problem("kettle.png", materials)
shape = shape_x, shape_y = outline.shape[:2]  # size of the domain

# TODO: clean this up and figure out the proper stability condition (0.1 is bogus)
delta_x = 1.0
alpha = k / c
delta_t = (delta_x ** 2) / (4 * alpha.max()) * 0.1  # numerical stability constraint

k = utils.convert_to_field(k)
cx = utils.convert_to_field((0.5 / c * delta_t) / (delta_x ** 2))
s = utils.convert_to_field(s)

# We are using 0 as boundary/initial conditions (sources are added later)
t = ti.field(dtype=ti.float32, shape=shape)
t_next = ti.field(dtype=ti.float32, shape=t.shape)


@ti.kernel
def diffusion(t: ti.template(), t_next: ti.template()):
    for i, j in t:
        if (0 < i < shape_x - 1) and (0 < j < shape_y - 1):
            if s[i, j] > 0:
                t_next[i, j] = s[i, j]
            else:

                t_next[i, j] = t[i, j] + cx[i, j] * (
                    (k[i + 1, j] + k[i, j]) * (t[i + 1, j] - t[i, j])
                    - (k[i - 1, j] + k[i, j]) * (t[i, j] - t[i - 1, j])
                    + (k[i, j + 1] + k[i, j]) * (t[i, j + 1] - t[i, j])
                    - (k[i, j - 1] + k[i, j]) * (t[i, j] - t[i, j - 1])
                )


scene = utils.Renderer(outline)
gui = ti.GUI("Heatsim", res=shape, fast_gui=True)

steps_per_frame = 1000
while not gui.get_event(ti.GUI.ESCAPE):
    chrono = utils.Chrono()

    for _ in range(steps_per_frame):
        diffusion(t, t_next)
        t, t_next = t_next, t
    chrono.log("diffusion")

    if gui.is_pressed(ti.GUI.LMB):
        mouse_x, mouse_y = gui.get_cursor_pos()
        t[int(mouse_x * shape_x), int(mouse_y * shape_y)] += 1.0
    chrono.log("mouse")

    img = scene.render(t)
    chrono.log("render")

    gui.set_image(img)
    chrono.log("set_image")

    gui.show()
    chrono.log("show")

    # trying to calibrate for optimal frame rate (tune down, only)
    fps = int(1.0 / chrono.elapsed("set_image"))  # show() may wait for a frame
    if fps < gui.fps_limit:
        steps_per_frame = max(1, int(0.95 * steps_per_frame))
        print(f"Steps per frame: {steps_per_frame}")

    # print(chrono)
