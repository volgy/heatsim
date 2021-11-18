#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions for numerical simulations"""

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
import matplotlib.cm as cm
from time import time
import taichi as ti

class Chrono:
    """Simple benchmarking tool (time log)"""

    def __init__(self):
        self.stamps = []
        self.log("start")

    def log(self, event):
        self.stamps.append((event, time()))

    def __repr__(self):
        total = self.stamps[-1][1] - self.stamps[0][1]
        tokens = []
        tokens.append(f"Total: {1000.0 * total:.1f} ms")
        for (_, start), (event, end) in zip(self.stamps[:-1], self.stamps[1:]):
            tokens.append(f"\n\t{event}: {1000.0 * (end - start):.1f} ms ")
        return "".join(tokens)


def load_problem(filename, dtype=np.float32):
    """Load a problem from a PNG file.

    The PNG file is expected to have the following R, G, B channels:
    - R: the diffusivity parameter of the material
    - G: the sources of heat
    - B: the outline (background) image

    Returns the following (normalized if dtype=float) numpy arrays:
    (diffusivity, sources, outline)
    """
    problem = ti.imread(filename)

    diffusivity = np.asarray(problem[:, :, 0], dtype)
    sources = np.asarray(problem[:, :, 1], dtype)
    outline = np.asarray(problem[:, :, 2], dtype)

    if np.issubdtype(dtype, np.floating):
        diffusivity /= 255.0
        sources /= 255.0
        outline /= 255.0

    return diffusivity, sources, outline


def convert_to_field(arr, dtype=ti.float32):
    """Convert a numpy array to a taichi field"""
    field = ti.field(dtype, arr.shape)
    field.from_numpy(arr)
    return field


@ti.kernel
def _render(
    background: ti.template(),
    field: ti.template(),
    lut: ti.template(),
    img: ti.template()):


    for i, j in field:
        idx = 0
        idx = min(max(0, int(field[i, j] * 255)), 255)
        if any(background[i, j] > 0):
            img[i, j] = background[i, j]
        else:
            img[i, j] = lut[idx]


class Renderer:
    """Fast field rendering with outline/background and colormaps.

    background is a numpy array of shape (size_x, size_y)
    """

    def __init__(self, background, cmap="inferno"):
        self.lut = ti.Vector.field(n=3, dtype=ti.uint8, shape=(256,))
        colormap = cm.get_cmap(cmap)
        for i in range(256):
            self.lut[i] = colormap(i/255.0, bytes=True)[:3]


        shape = background.shape
        self.background = ti.Vector.field(n=3, dtype=ti.uint8, shape=shape)
        self.background.from_numpy(
            np.repeat(255 * background[..., np.newaxis], 3, axis=2))

        self.img = ti.Vector.field(n=3, dtype=ti.uint8, shape=shape)


    def render(self, field):
        _render(self.background, field, self.lut, self.img)
        return self.img
