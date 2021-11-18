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

#ti.init(arch=ti.gpu)
ti.aot.start_recording('debug.yml')
ti.init(arch=ti.cc)
import utils

alpha, sources, outline = utils.load_problem("kettle.png")
size_x, size_y = alpha.shape[:2] # size of the domain

alpha = 255.0 * alpha # scaling (mm2/s)
delta_x = 0.25  # scaling (mm/px)
delta_t = (delta_x ** 2)/(4 * alpha.max()) # numerical stability constraint

gamma = utils.convert_to_field((alpha * delta_t) / (delta_x ** 2))
sources = utils.convert_to_field(sources)

# We are using 0 as boundary/initial conditions (sources are added later)
t = ti.field(dtype=ti.float32, shape=(size_x, size_y))
t_next = ti.field(dtype=ti.float32, shape=t.shape)

@ti.kernel
def diffusion(t: ti.template(), t_next: ti.template()):
    for i, j in t:
        if (0 < i < size_x - 1) and (0 < j < size_y - 1):
            if sources[i, j] > 0:
                t_next[i, j] = sources[i, j]
            else:
                t_next[i, j] = t[i, j] + gamma[i, j] * (
                    t[i+1, j] + t[i-1, j] + t[i, j+1] + t[i, j-1] - 4*t[i, j])


scene = utils.Renderer(outline)
gui = ti.GUI("Heatsim", res=(size_x, size_y), fast_gui=True)
while not gui.get_event(ti.GUI.ESCAPE):
    chrono = utils.Chrono()

    diffusion(t, t_next)
    t, t_next = t_next, t
    chrono.log("diffusion")


    if gui.is_pressed(ti.GUI.LMB):
        mouse_x, mouse_y = (gui.get_cursor_pos())
        t[int(mouse_x * size_x), int(mouse_y * size_y)] += 1.0
    chrono.log("mouse")

    img = scene.render(t)
    chrono.log("render")

    gui.set_image(img)
    chrono.log("set_image")

    gui.show()
    chrono.log("show")

    #print(chrono)
