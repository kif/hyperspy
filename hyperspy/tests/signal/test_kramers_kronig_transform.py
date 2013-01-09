# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.


import os

import numpy as np

from nose.tools import assert_true, assert_equal, assert_not_equal
from hyperspy.signals.eels import EELSSpectrum
from hyperspy.components import VolumePlasmonDrude, Lorentzian 

class Test1D:
    def setUp(self):
        ''' To test the kramers_kronig_transform we need two 
        EELSSpectrum instances. One will be the Drude bulk plasmon peak
        the other a synthetic fake-ZLP.
        '''
        # Create an empty spectrum
        s = EELSSpectrum({'data' : np.zeros((32,1024))})
        s.set_microscope_parameters(
                    beam_energy=300.0,
                    convergence_angle=14.0,
                    collection_angle=10.0)
        # Duplicate it
        z = s.deepcopy()
        ejeE = s.axes_manager.signal_axes[0]
        ejeE.scale = 0.02

        ejeE_z = z.axes_manager.signal_axes[0]
        ejeE_z.scale = ejeE.scale
        ejeE_z.offset = -10

        vpm = VolumePlasmonDrude()
        zlp = Lorentzian()

        zlp.A.value = 1.0
        zlp.gamma.value = 0.2
        zlp.centre.value = 0.0

        rnd=np.random.random
        ij=s.axes_manager
        vpm.intensity.value = 1.0
        vpm.plasmon_linewidth.value = 2.0

        for i in enumerate(s):
            vpm.plasmon_energy.value = 10 + (rnd() - 0.5) * 5
            # The ZLP
            s.data[ij.coordinates] = vpm.function(ejeE.axis)
            z.data[ij.coordinates] = zlp.function(ejeE_z.axis)
        
        self.signal = {'ELF': s, 'ZLP': z}
        
    def test_01(self):
        ''' The kramers kronig transform method applied to the signal we
        have just designed above will return the CDF for the Drude bulk
        plasmon. Hopefully, we recover the signal by inverting the CDF.
        '''
        items = self.signal
        elf = items['ELF']
        zlp = items['ZLP']
        eps=elf.kramers_kronig_transform(zlp)
        np.allclose(np.imag(-1/eps.data),elf.data,rtol=1)
        
        
