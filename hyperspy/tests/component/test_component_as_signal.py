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

from nose.tools import assert_true, assert_equal, assert_not_equal
from hyperspy.signals.spectrum import Spectrum
from hyperspy.hspy import create_model
from hyperspy.components import Gaussian


class TestComponentAsSignal:
    def setUp(self):
        g1 = Gaussian()
        g2 = Gaussian()
        s = Spectrum(np.arange(10000).reshape(10,10,100))
        m = create_model(s)
        m.append(g1)
        m.append(g2)
        g1.centre.value = 50
        g1.centre.assign_current_value_to_all()
        g1.A.value = 10
        g1.A.assign_current_value_to_all()
        g2.centre.value = 20
        g2.centre.assign_current_value_to_all()
        g2.A.value = 20
        g2.A.assign_current_value_to_all()
        self.g1 = g1
        self.g2 = g2
        self.model = m
        
    def test_component_as_signal1(self):
        m = self.model
        g1 = self.g1
        g2 = self.g2
        s_g1 = g1.component_as_signal()
        g1_data = g1.function(m.axes_manager.signal_axes[0].axis) 
        assert_true(np.all(s_g1[0][0] == g1_data))
