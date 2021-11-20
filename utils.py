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
from numpy.core.numerictypes import obj2sctype
import taichi as ti


class Chrono:
    """Simple benchmarking tool (time log)"""

    def __init__(self):
        self.stamps = []
        self.log("start")

    def log(self, event):
        self.stamps.append((event, time()))

    def elapsed(self, end=None, start=None):
        """Returns the elapsed time between two events.

        Default is the first and last events.
        """

        def find_stamp(event):
            for e, t in self.stamps:
                if e == event:
                    return t
            raise ValueError(f"Event {event} not found")

        if start is None:
            t_start = self.stamps[0][1]
        else:
            t_start = find_stamp(start)
        if end is None:
            t_end = self.stamps[-1][1]
        else:
            t_end = find_stamp(end)

        return t_end - t_start

    def __repr__(self):
        total = self.stamps[-1][1] - self.stamps[0][1]
        tokens = []
        tokens.append(f"Total: {1000.0 * total:.1f} ms")
        for (_, start), (event, end) in zip(self.stamps[:-1], self.stamps[1:]):
            tokens.append(f"\n\t{event}: {1000.0 * (end - start):.1f} ms ")
        return "".join(tokens)


def load_problem(filename, materials):
    """Load a problem from a PNG file.

    The PNG file is expected to be a gray-scale image (otherwise the first
    channel is used). The color values represent different material types,
    which are mapped according to the materials parameter.

    Returns the following numpy arrays:
    (c, k, s, outline)
    """
    problem = ti.imread(filename)[:, :, 0]

    objects = np.unique(problem)
    missing_materials = set(objects) - set(materials.keys())
    if missing_materials:
        raise ValueError(f"No materials are defined for: {missing_materials}")

    missing_objects = set(materials.keys()) - set(objects)
    if missing_objects:
        print(f"warning: no objects found for: {missing_objects}")

    lut_c = np.zeros(256, dtype=np.float32)
    lut_k = np.zeros(256, dtype=np.float32)
    lut_s = np.zeros(256, dtype=np.float32)

    for material, spec in materials.items():
        lut_c[material] = spec["c"]
        lut_k[material] = spec["k"]
        lut_s[material] = spec["s"]

    c = lut_c[problem]
    k = lut_k[problem]
    s = lut_s[problem]

    # Poor man's edge detector
    outline = np.asarray(
        (
            np.diff(problem, axis=0, prepend=0) ** 2
            + np.diff(problem, axis=1, prepend=0) ** 2
        )
        > 0,
        dtype=np.uint8,
    )

    return c, k, s, outline


def convert_to_field(arr, dtype=ti.float32):
    """Convert a numpy array to a taichi field"""
    field = ti.field(dtype, arr.shape)
    field.from_numpy(arr)
    return field


@ti.data_oriented
class Renderer:
    """Fast field rendering with outline/background and colormaps.

    background is a numpy array of shape (size_x, size_y)
    """

    def __init__(self, background, cmap="inferno"):
        self.lut = ti.Vector.field(n=3, dtype=ti.uint8, shape=(256,))
        colormap = cm.get_cmap(cmap)
        for i in range(256):
            self.lut[i] = colormap(i / 255.0, bytes=True)[:3]

        shape = background.shape
        self.background = ti.Vector.field(n=3, dtype=ti.uint8, shape=shape)
        self.background.from_numpy(
            np.repeat(255 * background[..., np.newaxis], 3, axis=2)
        )

        self.img = ti.Vector.field(n=3, dtype=ti.uint8, shape=shape)

    def render(self, field):
        self._render(field)
        return self.img

    @ti.kernel
    def _render(self, field: ti.template()):
        for i, j in field:
            idx = 0
            idx = min(max(0, int(field[i, j] * 255)), 255)
            if any(self.background[i, j] > 0):
                self.img[i, j] = self.background[i, j]
            else:
                self.img[i, j] = self.lut[idx]
