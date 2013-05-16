# Copyright 2007-2012 The Hyperspy developers
#
# This file is part of Hyperspy.
#
# Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Hyperspy. If not, see <http://www.gnu.org/licenses/>.


import os

import numpy as np

from nose.tools import (assert_true,
                        assert_false,
                        assert_equal,
                        assert_not_equal,
                        raises)
from hyperspy.components import Gaussian

class TestSetParameters:
    def setUp(self):
        self.gaussian = Gaussian()
        
    def test_set_parameters_not_free1(self):
        g = self.gaussian
        g.set_parameters_not_free()
        free_parameters = len(g.free_parameters)
        assert_equal(free_parameters, 0)

    def test_set_parameters_not_free2(self):
        g = self.gaussian
        g.set_parameters_not_free(parameter_name_list=['area'])
        free_parameters = len(g.free_parameters)
        parameters = len(g.parameters) - 1
        assert_equal(free_parameters, parameters)

    def test_set_parameters_free1(self):
        g = self.gaussian
        g.area.free = False
        g.set_parameters_free()
        free_parameters = len(g.free_parameters)
        parameters = len(g.parameters)
        assert_equal(free_parameters, parameters)

    def test_set_parameters_free2(self):
        g = self.gaussian
        g.area.free = False
        g.centre.free = False
        g.sigma.free = False
        g.set_parameters_free(parameter_name_list=['area'])
        free_parameters = len(g.free_parameters)
        parameters = len(g.parameters) - 2
        assert_equal(free_parameters, parameters)
