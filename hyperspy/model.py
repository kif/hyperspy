# -*- coding: utf-8 -*-
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

import copy
import os
import tempfile

import numpy as np
import numpy.linalg

import traits.api as t
import scipy.odr as odr
from scipy.optimize import (leastsq,fmin, fmin_cg, fmin_ncg, fmin_bfgs,
    fmin_cobyla, fmin_l_bfgs_b, fmin_tnc, fmin_powell)

from hyperspy import messages
import hyperspy.drawing.spectrum
from hyperspy.drawing.utils import on_figure_window_close
from hyperspy.misc import progressbar, utils
from hyperspy.signals.eels import EELSSpectrum, Spectrum
from hyperspy.defaults_parser import preferences
from hyperspy.axes import generate_axis
from hyperspy.exceptions import WrongObjectError
from hyperspy.decorators import interactive_range_selector
from hyperspy.misc.mpfit.mpfit import mpfit

from hyperspy.gui.tools import ComponentFit

import matplotlib.pyplot as plt

class Model(list):
    """Build and fit a model
    
    Parameters
    ----------
    spectrum : an Spectrum (or any Spectrum subclass) instance
    """
    
    _firstimetouch = True

    def __init__(self, spectrum):
        self.convolved = False
        self.spectrum = spectrum
        self.axes_manager = self.spectrum.axes_manager
        self.axis = self.axes_manager.signal_axes[0]
        self.axes_manager.connect(self.charge)
         
        self.free_parameters_boundaries = None
        # TODO: model cube should dissapear or at least be an option
        self.model_cube = np.zeros(self.spectrum.data.shape, 
                                   dtype = 'float')
        self.model_cube[:] = np.nan
        self.channel_switches=np.array([True] * len(self.axis.axis))
        self._low_loss = None
        self._plot = None


    @property
    def spectrum(self):
        return self._spectrum
        
    @spectrum.setter
    def spectrum(self, value):
        if isinstance(value, Spectrum):
            self._spectrum = value
        else:
            raise WrongObjectError(str(type(value)), 'Spectrum')
                    
    @property
    def low_loss(self):
        return self._low_loss
        
    @low_loss.setter
    def low_loss(self, value):
        if value is not None:
            if (value.axes_manager.navigation_shape != 
                self.spectrum.axes_manager.navigation_shape):
                    raise ValueError('The low-loss does not have '
                        'the same navigation dimension as the '
                        'core-loss')
            self._low_loss = value
            self.set_convolution_axis()
            self.convolved = True
        else:
            self._low_loss = value
            self.convolution_axis = None
            self.convolved = False

        
    # Extend the list methods to call the _touch when the model is modified
    def append(self, object):
        object.create_arrays(self.axes_manager.navigation_shape)
        object.set_axes(self.axes_manager)
        list.append(self,object)
        self._touch()
    
    def insert(self, object):
        object.create_arrays(self.axes_manager.navigation_shape)
        object.set_axes(self.axes_manager)
        list.insert(self,object)
        self._touch()
   
    def extend(self, iterable):
        for object in iterable:
            object.create_arrays(self.axes_manager.navigation_shape)
            object.set_axes(self.axes_manager)
        list.extend(self,iterable)
        self._touch()
                
    def __delitem__(self, object):
        list.__delitem__(self,object)
        self._touch()
    
    def remove(self, object, touch=True):
        list.remove(self,object)
        if touch is True:
            self._touch() 

    def _touch(self):
        """Run model setup tasks
        
        This function is called everytime that we add or remove components
        from the model.
        """
        if self._get_auto_update_plot() is True:
            self._connect_parameters2update_plot()
        
    __touch = _touch
    
    def set_convolution_axis(self):
        """
        Creates an axis to use to generate the data of the model in the precise
        scale to obtain the correct axis and origin after convolution with the
        lowloss spectrum.
        """
        ll_axis = self.low_loss.axes_manager.signal_axes[0]
        dimension = self.axis.size + ll_axis.size - 1
        step = self.axis.scale
        knot_position = ll_axis.size - ll_axis.value2index(0) - 1
        self.convolution_axis = generate_axis(self.axis.offset, step, 
        dimension, knot_position)
                
    def _connect_parameters2update_plot(self):   
        for component in self:
            component.connect(self.update_plot)
            for parameter in component.parameters:
                if self.spectrum._plot is not None:
                    parameter.connect(self.update_plot)
    
    def _disconnect_parameters2update_plot(self):
        for component in self:
            component.disconnect(self.update_plot)
            for parameter in component.parameters:
                parameter.disconnect(self.update_plot)
    

    def generate_data_from_model(self, out_of_range_to_nan=True):
        """Generate a SI with the current model
        
        The SI is stored in self.model_cube
        """
        maxval = self.axes_manager.navigation_size
        if maxval > 0:
            pbar = progressbar.progressbar(maxval=maxval)
        i = 0
        for index in self.axes_manager:
            self.charge(only_fixed=False)
            self.model_cube[self.axes_manager._getitem_tuple][
            self.channel_switches] = self.__call__(
                non_convolved=not self.convolved, onlyactive=True)
            if out_of_range_to_nan is True:
                self.model_cube[self.axes_manager._getitem_tuple][
                self.channel_switches == False] = np.nan
            i += 1
            if maxval > 0:
                pbar.update(i)
        pbar.finish()
        
    def _get_auto_update_plot(self):
        if self._plot is not None and self._plot.is_active() is True:
            return True
        else:
            return False
            
# TODO: port it                    
#    def generate_chisq(self, degrees_of_freedom = 'auto') :
#        if self.spectrum.variance is None:
#            self.spectrum.estimate_variance()
#        variance = self.spectrum.variance[self.channel_switches]
#        differences = (self.model_cube - self.spectrum.data)[self.channel_switches]
#        self.chisq = np.sum(differences**2 / variance, 0)
#        if degrees_of_freedom == 'auto':
#            self.red_chisq = self.chisq / \
#            (np.sum(np.ones(self.spectrum.energydimension)[self.channel_switches]) \
#            - len(self.p0) -1)
#            print "Degrees of freedom set to auto"
#            print "DoF = ", len(self.p0)
#        elif type(degrees_of_freedom) is int :
#            self.red_chisq = self.chisq / \
#            (np.sum(np.ones(self.spectrum.energydimension)[self.channel_switches]) \
#            - degrees_of_freedom -1)
#        else:
#            print "degrees_of_freedom must be on interger type."
#            print "The red_chisq could not been calculated"

    def _set_p0(self):
        p0 = None
        for component in self:
            if component.active:
                for param in component.free_parameters:
                    if p0 is not None:
                        p0 = (p0 + [param.value,] 
                        if not isinstance(param.value, list) 
                        else p0 + param.value)
                    else:
                        p0 = ([param.value,] 
                        if not isinstance(param.value, list) 
                        else param.value)
        self.p0 = tuple(p0)
    
    def set_boundaries(self):
        """Generate the boundary list.
        
        Necessary before fitting with a boundary awared optimizer
        """
        self.free_parameters_boundaries = []
        for component in self:
            if component.active:
                for param in component.free_parameters:
                    if param._number_of_elements == 1:
                        self.free_parameters_boundaries.append((
                        param._bounds))
                    else:
                        self.free_parameters_boundaries.extend((
                        param._bounds))
                        
    def set_mpfit_parameters_info(self):
        self.mpfit_parinfo = []
        for component in self:
            if component.active:
                for param in component.free_parameters:
                    if param._number_of_elements == 1:
                        limited = [False,False]
                        limits = [0,0]
                        if param.bmin is not None:
                            limited[0] = True
                            limits[0] = param.bmin
                        if param.bmax is not None:
                            limited[1] = True
                            limits[1] = param.bmax
                        self.mpfit_parinfo.append(
                        {'limited' : limited,
                         'limits' : limits})

    def set(self):
        """ Store the parameters of the current coordinates into the 
        parameters array.
        
        If the parameters array has not being defined yet it creates it filling 
        it with the current parameters."""
        for component in self:
            component.store_current_parameters_in_map(
                                                 self.axes_manager.indexes)

    def charge(self, only_fixed=False):
        """Charge the parameters for the current spectrum from the parameters 
        array
        
        Parameters
        ----------
        only_fixed : bool
            If True, only the fixed parameters will be charged.
            
        """
        switch_aap = (False != self._get_auto_update_plot())
        if switch_aap is True:
            self._disconnect_parameters2update_plot()
        for component in self:
            component.charge_value_from_map(self.axes_manager.indexes,
                                            only_fixed=only_fixed)
        if switch_aap is True:
            self._connect_parameters2update_plot()
            self.update_plot()

    def update_plot(self):
        if self.spectrum._plot is not None:
            try:
                for line in self.spectrum._plot.signal_plot.ax_lines:
                        line.update()
            except:
                self._disconnect_parameters2update_plot()
                
    def _charge_p0(self, p_std = None):
        """Charge the free data for the current coordinates (x,y) from the
        p0 array.
        
        Parameters
        ----------
        p_std : array
            array containing the corresponding standard deviation
        """
        comp_p_std = None
        counter = 0
        for component in self: # Cut the parameters list
            if component.active is True:
                if p_std is not None:
                    comp_p_std = p_std[counter: counter + component._nfree_param]
                component.charge(
                self.p0[counter: counter + component._nfree_param], 
                comp_p_std, onlyfree = True)
                counter += component._nfree_param

    # Defines the functions for the fitting process -------------------------
    def _model2plot(self, axes_manager, out_of_range2nans=True):
        old_axes_manager = None
        if axes_manager is not self.axes_manager:
            old_axes_manager = self.axes_manager
            self.axes_manager = axes_manager
            self.charge()
        s = self.__call__(non_convolved=False, onlyactive=True)
        if old_axes_manager is not None:
            self.axes_manager = old_axes_manager
            self.charge()
        if out_of_range2nans is True:
            ns = np.zeros((self.axis.axis.shape))
            ns[:] = np.nan
            ns[self.channel_switches] = s
            s = ns
        return s
    
    def __call__(self, non_convolved=False, onlyactive=False) :
        """Returns the corresponding model for the current coordinates
        
        Parameters
        ----------
        non_convolved : bool
            If True it will return the deconvolved model
        only_active : bool
            If True, only the active components will be used to build the model.
            
        cursor: 1 or 2
        
        Returns
        -------
        numpy array
        """
            
        if self.convolved is False or non_convolved is True:
            axis = self.axis.axis[self.channel_switches]
            sum_ = np.zeros(len(axis))
            if onlyactive is True:
                for component in self: # Cut the parameters list
                    if component.active:
                        np.add(sum_, component.function(axis),
                        sum_)
                return sum_
            else:
                for component in self: # Cut the parameters list
                    np.add(sum_, component.function(axis),
                     sum_)
                return sum_

        else: # convolved
            counter = 0
            sum_convolved = np.zeros(len(self.convolution_axis))
            sum_ = np.zeros(len(self.axis.axis))
            for component in self: # Cut the parameters list
                if onlyactive :
                    if component.active:
                        if component.convolved:
                            np.add(sum_convolved,
                            component.function(
                            self.convolution_axis), sum_convolved)
                        else:
                            np.add(sum_,
                            component.function(self.axis.axis), sum_)
                        counter+=component._nfree_param
                else :
                    if component.convolved:
                        np.add(sum_convolved,
                        component.function(self.convolution_axis),
                        sum_convolved)
                    else:
                        np.add(sum_, component.function(self.axis.axis),
                        sum_)
                    counter+=component._nfree_param
            to_return = sum_ + np.convolve(
                self.low_loss(self.axes_manager), 
                sum_convolved, mode="valid")
            to_return = to_return[self.channel_switches]
            return to_return

    # TODO: the way it uses the axes
    def _set_signal_range_in_pixels(self, i1=None, i2=None):
        """Use only the selected spectral range in the fitting routine.
        
        Parameters
        ----------
        i1 : Int
        i2 : Int
        
        Notes
        -----
        To use the full energy range call the function without arguments.
        """

        self.backup_channel_switches = copy.copy(self.channel_switches)
        self.channel_switches[:] = False
        self.channel_switches[i1:i2] = True
        if self._get_auto_update_plot() is True:
            self.update_plot()
            
    @interactive_range_selector   
    def set_signal_range(self, x1=None, x2=None):
        """Use only the selected spectral range defined in its own units in the 
        fitting routine.
        
        Parameters
        ----------
        E1 : None or float
        E2 : None or float
        
        Notes
        -----
        To use the full energy range call the function without arguments.
        """
        i1, i2 = self.axis.value2index(x1), self.axis.value2index(x2)
        self._set_signal_range_in_pixels(i1, i2)

    def _remove_signal_range_in_pixels(self, i1=None, i2=None):
        """Removes the data in the given range from the data range that 
        will be used by the fitting rountine
        
        Parameters
        ----------
        x1 : None or float
        x2 : None or float
        """
        self.channel_switches[i1:i2] = False
        if self._get_auto_update_plot() is True:
            self.update_plot()

    @interactive_range_selector    
    def remove_signal_range(self, x1=None, x2=None):
        """Removes the data in the given range from the data range that 
        will be used by the fitting rountine
        
        Parameters
        ----------
        x1 : None or float
        x2 : None or float
        
        """
        i1, i2 = self.axis.value2index(x1), self.axis.value2index(x2)
        self._remove_signal_range_in_pixels(i1, i2)
        
    def reset_signal_range(self):
        '''Resets the data range'''
        self._set_signal_range_in_pixels()
    
    def _add_signal_range_in_pixels(self, i1=None, i2=None):
        """Adds the data in the given range from the data range that 
        will be used by the fitting rountine
        
        Parameters
        ----------
        x1 : None or float
        x2 : None or float
        """
        self.channel_switches[i1:i2] = True
        if self._get_auto_update_plot() is True:
            self.update_plot()

    @interactive_range_selector    
    def add_signal_range(self, x1=None, x2=None):
        """Adds the data in the given range from the data range that 
        will be used by the fitting rountine
        
        Parameters
        ----------
        x1 : None or float
        x2 : None or float
        
        """
        i1, i2 = self.axis.value2index(x1), self.axis.value2index(x2)
        self._add_signal_range_in_pixels(i1, i2)
        
    def reset_the_signal_range(self):
        self.channel_switches[:] = True
        if self._get_auto_update_plot() is True:
            self.update_plot()

    def _model_function(self,param):

        if self.convolved is True:
            counter = 0
            sum_convolved = np.zeros(len(self.convolution_axis))
            sum = np.zeros(len(self.axis.axis))
            for component in self: # Cut the parameters list
                if component.active is True:
                    if component.convolved is True:
                        np.add(sum_convolved, component(param[\
                        counter:counter+component._nfree_param],
                        self.convolution_axis), sum_convolved)
                    else:
                        np.add(sum, component(param[counter:counter + \
                        component._nfree_param], self.axis.axis), sum)
                    counter+=component._nfree_param

            return (sum + np.convolve(self.low_loss(self.axes_manager), 
                                      sum_convolved,mode="valid"))[
                                      self.channel_switches]

        else:
            axis = self.axis.axis[self.channel_switches]
            counter = 0
            first = True
            for component in self: # Cut the parameters list
                if component.active is True:
                    if first is True:
                        sum = component(param[counter:counter + \
                        component._nfree_param],axis)
                        first = False
                    else:
                        sum += component(param[counter:counter + \
                        component._nfree_param], axis)
                    counter += component._nfree_param
            return sum

    def _jacobian(self,param, y, weights=None):
        if self.convolved is True:
            counter = 0
            grad = np.zeros(len(self.axis.axis))
            for component in self: # Cut the parameters list
                if component.active:
                    component.charge(param[counter:counter + \
                    component._nfree_param] , onlyfree = True)
                    if component.convolved:
                        for parameter in component.free_parameters :
                            par_grad = np.convolve(
                            parameter.grad(self.convolution_axis), 
                            self.low_loss(self.axes_manager), 
                            mode="valid")
                            if parameter._twins:
                                for parameter in parameter._twins:
                                    np.add(par_grad, np.convolve(
                                    parameter.grad(
                                    self.convolution_axis), 
                                    self.low_loss(self.axes_manager), 
                                    mode="valid"), par_grad)
                            grad = np.vstack((grad, par_grad))
                        counter += component._nfree_param
                    else:
                        for parameter in component.free_parameters :
                            par_grad = parameter.grad(self.axis.axis)
                            if parameter._twins:
                                for parameter in parameter._twins:
                                    np.add(par_grad, parameter.grad(
                                    self.axis.axis), par_grad)
                            grad = np.vstack((grad, par_grad))
                        counter += component._nfree_param
            if weights is None:
                return grad[1:, self.channel_switches]
            else:
                return grad[1:, self.channel_switches] * weights
        else:
            axis = self.axis.axis[self.channel_switches]
            counter = 0
            grad = axis
            for component in self: # Cut the parameters list
                if component.active:
                    component.charge(param[counter:counter + \
                    component._nfree_param] , onlyfree = True)
                    for parameter in component.free_parameters :
                        par_grad = parameter.grad(axis)
                        if parameter._twins:
                            for parameter in parameter._twins:
                                np.add(par_grad, parameter.grad(
                                axis), par_grad)
                        grad = np.vstack((grad, par_grad))
                    counter += component._nfree_param
            if weights is None:
                return grad[1:,:]
            else:
                return grad[1:,:] * weights
        
    def _function4odr(self,param,x):
        return self._model_function(param)
    
    def _jacobian4odr(self,param,x):
        return self._jacobian(param, x)
        
    def calculate_p_std(self, p0, method, *args):
        print "Estimating the standard deviation"
        f = self._poisson_likelihood_function if method == 'ml' \
        else self._errfunc2
        hess = approx_hessian(p0,f,*args)
        ihess = np.linalg.inv(hess)
        p_std = np.sqrt(1./np.diag(ihess))
        return p_std

    def _poisson_likelihood_function(self,param,y, weights=None):
        """Returns the likelihood function of the model for the given
        data and parameters
        """
        mf = self._model_function(param)
        return -(y*np.log(mf) - mf).sum()

    def _gradient_ml(self,param, y, weights=None):
        mf = self._model_function(param)
        return -(self._jacobian(param, y)*(y/mf - 1)).sum(1)


    def _errfunc(self,param, y, weights=None):
        errfunc = self._model_function(param) - y
        if weights is None:
            return errfunc
        else:
            return errfunc * weights
    def _errfunc2(self,param, y, weights=None):
        if weights is None:
            return ((self._errfunc(param, y))**2).sum()
        else:
            return ((weights * self._errfunc(param, y))**2).sum()

    def _gradient_ls(self,param, y, weights=None):
        gls =(2*self._errfunc(param, y, weights) * 
        self._jacobian(param, y)).sum(1)
        return gls
        
    def _errfunc4mpfit(self, p, fjac=None, x=None, y=None,
        weights = None):
        if fjac is None:
            errfunc = self._model_function(p) - y
            if weights is not None:
                errfunc *= weights
            jacobian = None
            status = 0
            return [status, errfunc]
        else:
            return [0, self._jacobian(p,y).T]
        
    def fit(self, fitter=None, method='ls', grad=False, weights=None,
            bounded=False, ext_bounding=False, update_plot=False, 
            **kwargs):
        """Fits the model to the experimental data
        
        Parameters
        ----------
        fitter : {None, "leastsq", "odr", "mpfit", "fmin"}
            The optimizer to perform the fitting. If None the fitter
            defined in the Preferences is used. leastsq is the most 
            stable but it does not support bounding. mpfit supports
            bounding. fmin is the only one that supports 
            maximum likelihood estimation, but it is less robust than 
            the Levenberg–Marquardt based leastsq and mpfit, and it is 
            better to use it after one of them to refine the estimation.
        method : {'ls', 'ml'}
            Choose 'ls' (default) for least squares and 'ml' for 
            maximum-likelihood estimation. The latter only works with 
            fitter = 'fmin'.
        grad : bool
            If True, the analytical gradient is used if defined to 
            speed up the estimation. 
        weights : {None, True, numpy.array}
            If None, performs standard least squares. If True 
            performs weighted least squares where the weights are 
            calculated using spectrum.Spectrum.estimate_variance. 
            Alternatively, external weights can be supplied by passing
            a weights array of the same dimensions as the signal.
        ext_bounding : bool
            If True, enforce bounding by keeping the value of the 
            parameters constant out of the defined bounding area.
        bounded : bool
            If True performs bounded optimization if the fitter 
            supports it. Currently only mpfit support bounding. 
        update_plot : bool
            If True, the plot is updated during the optimization 
            process. It slows down the optimization but it permits
            to visualize the optimization progress. 
        
        **kwargs : key word arguments
            Any extra key word argument will be passed to the chosen
            fitter
            
        See Also
        --------
        multifit
            
        """
        if fitter is None:
            fitter = preferences.Model.default_fitter
        switch_aap = (update_plot != self._get_auto_update_plot())
        if switch_aap is True and update_plot is False:
            self._disconnect_parameters2update_plot()
            
        self.p_std = None
        self._set_p0()
        if ext_bounding:
            self._enable_ext_bounding()
        if grad is False :
            approx_grad = True
            jacobian = None
            odr_jacobian = None
            grad_ml = None
            grad_ls = None
        else :
            approx_grad = False
            jacobian = self._jacobian
            odr_jacobian = self._jacobian4odr
            grad_ml = self._gradient_ml
            grad_ls = self._gradient_ls
        if method == 'ml':
            weights = None
        if weights is True:
            if self.spectrum.variance is None:
                self.spectrum.estimate_variance()
            weights = 1. / np.sqrt(self.spectrum.variance.__getitem__(
            self.axes_manager._getitem_tuple)[self.channel_switches])
        elif weights is not None:
            weights = weights.__getitem__(
                self.axes_manager._getitem_tuple)[
                    self.channel_switches]
        args = (self.spectrum()[self.channel_switches], 
        weights)
        
        # Least squares "dedicated" fitters
        if fitter == "leastsq":
            output = \
            leastsq(self._errfunc, self.p0[:], Dfun = jacobian,
            col_deriv=1, args = args, full_output = True, **kwargs)
            
            self.p0 = output[0]
            var_matrix = output[1]
            # In Scipy 0.7 sometimes the variance matrix is None (maybe a 
            # bug?) so...
            if var_matrix is not None:
                self.p_std = np.sqrt(np.diag(var_matrix))
            self.fit_output = output
        
        elif fitter == "odr":
            modelo = odr.Model(fcn = self._function4odr, 
            fjacb = odr_jacobian)
            mydata = odr.RealData(self.axis.axis[self.channel_switches],
            self.spectrum()[self.channel_switches],
            sx = None,
            sy = (1/weights if weights is not None else None))
            myodr = odr.ODR(mydata, modelo, beta0=self.p0[:])
            myoutput = myodr.run()
            result = myoutput.beta
            self.p_std = myoutput.sd_beta
            self.p0 = result
            self.fit_output = myoutput
            
        elif fitter == 'mpfit':
            autoderivative = 1
            if grad is True:
                autoderivative = 0

            if bounded is True:
                self.set_mpfit_parameters_info()
            elif bounded is False:
                self.mpfit_parinfo = None
            m = mpfit(self._errfunc4mpfit, self.p0[:], 
                parinfo=self.mpfit_parinfo, functkw= {
                'y': self.spectrum()[self.channel_switches], 
                'weights' :weights}, autoderivative = autoderivative,
                quiet = 1)
            self.p0 = m.params
            self.p_std = m.perror
            self.fit_output = m
            
        else:          
        # General optimizers (incluiding constrained ones(tnc,l_bfgs_b)
        # Least squares or maximum likelihood
            if method == 'ml':
                tominimize = self._poisson_likelihood_function
                fprime = grad_ml
            elif method == 'ls':
                tominimize = self._errfunc2
                fprime = grad_ls
                        
            # OPTIMIZERS
            
            # Simple (don't use gradient)
            if fitter == "fmin" :
                self.p0 = fmin(
                    tominimize, self.p0, args = args, **kwargs)
            elif fitter == "powell" :
                self.p0 = fmin_powell(tominimize, self.p0, args = args, 
                **kwargs)
            
            # Make use of the gradient
            elif fitter == "cg" :
                self.p0 = fmin_cg(tominimize, self.p0, fprime = fprime,
                args= args, **kwargs)
            elif fitter == "ncg" :
                self.p0 = fmin_ncg(tominimize, self.p0, fprime = fprime,
                args = args, **kwargs)
            elif fitter == "bfgs" :
                self.p0 = fmin_bfgs(
                    tominimize, self.p0, fprime = fprime,
                    args = args, **kwargs)
            
            # Constrainded optimizers
            
            # Use gradient
            elif fitter == "tnc":
                if bounded is True:
                    self.set_boundaries()
                elif bounded is False:
                    self.self.free_parameters_boundaries = None
                self.p0 = fmin_tnc(tominimize, self.p0, fprime = fprime,
                args = args, bounds = self.free_parameters_boundaries, 
                approx_grad = approx_grad, **kwargs)[0]
            elif fitter == "l_bfgs_b":
                if bounded is True:
                    self.set_boundaries()
                elif bounded is False:
                    self.self.free_parameters_boundaries = None
                self.p0 = fmin_l_bfgs_b(tominimize, self.p0,
                    fprime=fprime, args=args, 
                    bounds=self.free_parameters_boundaries, 
                    approx_grad = approx_grad, **kwargs)[0]
            else:
                print \
                """
                The %s optimizer is not available.

                Available optimizers:
                Unconstrained:
                --------------
                Only least Squares: leastsq and odr
                General: fmin, powell, cg, ncg, bfgs

                Cosntrained:
                ------------
                tnc and l_bfgs_b
                """ % fitter
                
        
        if np.iterable(self.p0) == 0:
            self.p0 = (self.p0,)
        self._charge_p0(p_std = self.p_std)
        self.set()
        if ext_bounding is True:
            self._disable_ext_bounding()
        if switch_aap is True and update_plot is False:
            self._connect_parameters2update_plot()
            self.update_plot()            
                
    def multifit(self, mask=None, charge_only_fixed=False,
                 autosave=False, autosave_every=10, **kwargs):
        """Fit the data to the model at all the positions of the 
        navigation dimensions.        
        
        Parameters
        ----------
        
        mask : {None, numpy.array}
            To mask (do not fit) at certain position pass a numpy.array
            of type bool where True indicates that the data will not be
            fitted at the given position.
        charge_only_fixed : bool
            If True, only the fixed parameters values will be updated
            when changing the positon.
        autosave : bool
            If True, the result of the fit will be saved automatically
            with a frequency defined by autosave_every.
        autosave_every : int
            Save the result of fitting every given number of spectra.
        
        **kwargs : key word arguments
            Any extra key word argument will be passed to 
            the fit method. See the fit method documentation for 
            a list of valid arguments.
            
        See Also
        --------
        fit
            
        """
        
        if autosave is not False:
            fd, autosave_fn = tempfile.mkstemp(
                prefix = 'hyperspy_autosave-', 
                dir = '.', suffix = '.npz')
            os.close(fd)
            autosave_fn = autosave_fn[:-4]
            messages.information(
            "Autosaving each %s pixels to %s.npz" % (autosave_every, 
                                                     autosave_fn))
            messages.information(
            "When multifit finishes its job the file will be deleted")
        if mask is not None and \
        (mask.shape != tuple(self.axes_manager.navigation_shape)):
           messages.warning_exit(
           "The mask must be a numpy array of boolen type with "
           " the same shape as the navigation: %s" % 
           self.axes_manager.navigation_shape)
        masked_elements = 0 if mask is None else mask.sum()
        maxval=self.axes_manager.navigation_size - masked_elements
        if maxval > 0:
            pbar = progressbar.progressbar(maxval=maxval)
        if 'bounded' in kwargs and kwargs['bounded'] is True:
            if kwargs['fitter'] == 'mpfit':
                self.set_mpfit_parameters_info()
                kwargs['bounded'] = None
            elif kwargs['fitter'] in ("tnc", "l_bfgs_b"):
                self.set_boundaries()
                kwargs['bounded'] = None
            else:
                messages.information(
                "The chosen fitter does not suppport bounding."
                "If you require boundinig please select one of the "
                "following fitters instead: mpfit, tnc, l_bfgs_b")
                kwargs['bounded'] = False
        i = 0
        for index in self.axes_manager:
            if mask is None or not mask[index]:
                self.fit(**kwargs)
                i += 1
                if maxval > 0:
                    pbar.update(i)
            if autosave is True and i % autosave_every  == 0:
                self.save_parameters2file(autosave_fn)
        if maxval > 0:
            pbar.finish()
        if autosave is True:
            messages.information(
            'Deleting the temporary file %s pixels' % (
                autosave_fn + 'npz'))
            os.remove(autosave_fn + '.npz')

            
    def save_parameters2file(self, filename):
        """Save the parameters array in binary format
        
        Parameters
        ----------
        filename : str
        
        """
        kwds = {}
        i = 0
        for component in self:
            cname = component.name.lower().replace(' ', '_')
            for param in component.parameters:
                pname = param.name.lower().replace(' ', '_')
                kwds['%s_%s.%s' % (i, cname, pname)] = param.map
            i += 1
        np.savez(filename, **kwds)

    def load_parameters_from_file(self,filename):
        """Loads the parameters array from  a binary file written with 
        the 'save_parameters2file' function
        
        Paramters
        ---------
        filename : str
        
        """
        
        f = np.load(filename)
        i = 0
        for component in self: # Cut the parameters list
            cname = component.name.lower().replace(' ', '_')
            for param in component.parameters:
                pname = param.name.lower().replace(' ', '_')
                param.map = f['%s_%s.%s' % (i, cname, pname)]
            i += 1
                
        self.charge()
           
    def plot(self):
        """Plots the current spectrum to the screen and a map with a 
        cursor to explore the SI.
        
        """
        
        # If new coordinates are assigned
        self.spectrum.plot()
        _plot = self.spectrum._plot
        l1 = _plot.signal_plot.ax_lines[0]
        color = l1.line.get_color()
        l1.line_properties_helper(color, 'scatter')
        l1.set_properties()
        
        l2 = hyperspy.drawing.spectrum.SpectrumLine()
        l2.data_function = self._model2plot
        l2.line_properties_helper('blue', 'line')        
        # Add the line to the figure
        _plot.signal_plot.add_line(l2)
        l2.plot()
        on_figure_window_close(_plot.signal_plot.figure, 
                                self._disconnect_parameters2update_plot)
        self._plot = self.spectrum._plot
        self._connect_parameters2update_plot()
        
    def set_current_values_to(self, components_list=None, mask=None):
        if components_list is None:
            components_list = []
            for comp in self:
                if comp.active:
                    components_list.append(comp)
        for comp in components_list:
            for parameter in comp.parameters:
                parameter.set_current_value_to(mask = mask)
                
    def _enable_ext_bounding(self,components = None):
        """
        """
        if components is None :
            components = self
        for component in components:
            for parameter in component.parameters:
                parameter.ext_bounded = True
    def _disable_ext_bounding(self,components = None):
        """
        """
        if components is None :
            components = self
        for component in components:
            for parameter in component.parameters:
                parameter.ext_bounded = False
                
    def export_results(self, folder=None, format=None, save_std=False,
                       only_free=True, only_active=True):
        """Export the results of the parameters of the model to the desired
        folder.
        
        Parameters
        ----------
        folder : str or None
            The path to the folder where the file will be saved. If `None` the
            current folder is used by default.
        format : str
            The format to which the data will be exported. It must be the
            extension of any format supported by Hyperspy. If None, the default
            format for exporting as defined in the `Preferences` will be used.
        save_std : bool
            If True, also the standard deviation will be saved.
        only_free : bool
            If True, only the value of the parameters that are free will be
            exported.
        only_active : bool
            If True, only the value of the active parameters will be exported.
            
        Notes
        -----
        The name of the files will be determined by each the Component and
        each Parameter name attributes. Therefore, it is possible to customise
        the file names modify the name attributes.
              
        """
        for component in self:
            if only_active is False or component.active is True:
                component.export(folder=folder, format=format,
                                 save_std=save_std, only_free=only_free)
                                 
    def plot_results(self, only_free=True, only_active = True):
        """Plot the value of the parameters of the model
        
        Parameters
        ----------

        only_free : bool
            If True, only the value of the parameters that are free will be
            plotted.
        only_active : bool
            If True, only the value of the active parameters will be plotted.
            
        Notes
        -----
        The name of the files will be determined by each the Component and
        each Parameter name attributes. Therefore, it is possible to customise
        the file names modify the name attributes.
              
        """
        for component in self:
            if only_active is False or component.active is True:
                component.plot(only_free=only_free)
                
    def print_current_values(self, only_free=True):
        """Print the value of each parameter of the model.
        
        Parameters
        ----------
        only_free : bool
            If True, only the value of the parameters that are free will
             be printed.
             
        """
        print "Components\tParameter\tValue"
        for component in self:
            if component.active is True:
                if component.name:
                    print(component.name)
                else:
                    print(component._id_name)
                parameters = component.free_parameters if only_free \
                    else component.parameters
                for parameter in parameters:
                    if not hasattr(parameter.value, '__iter__'):
                        print("\t\t%s\t%f" % (
                            parameter.name, parameter.value))
                            

    def fit_component(self, component, signal_range="interactive",
            estimate_parameters=True, fit_independent=False, **kwargs):
        """Fit just the given component in the given signal range.


        This method is useful to obtain starting parameters for the 
        components. Any keyword arguments are passed to the fit method.


        Parameters
        ----------
        component: component instance
            The component must be in the model, otherwise an exception 
            is raised.
        signal_range: {'interactive', (left_value, right_value), None}
            If 'interactive' the signal range is selected using the span
             selector on the spectrum plot. The signal range can also 
             be manually specified by passing a tuple of floats. If None
              the current signal range is used.
        estimate_parameters: bool, default True
            If True will check if the component has an 
            estimate_parameters function, and use it to estimate the
            parameters in the component.
        fit_independent: bool, default False
            Fits component independently of the other components in the
            model.

        Example
        -------
        Signal range set interactivly
        >>> g1 = components.Gaussian()
        >>> m.append(g1)
        >>> m.fit_component(g1)
        
        Signal range set through direct input
        >>> m.fit_component(g1, signal_range=(50,100))
        """
        
        cf = ComponentFit(self, component, signal_range,
                estimate_parameters, fit_independent, **kwargs)
        if signal_range == "interactive":
            cf.edit_traits()
        else:
            cf.apply()
        
    def fit_all_components(self):
        for component in self:
            self.fit_component(component, signal_range="interactive")

    def plot_components(self, components=None, plot_combined=False,
            plot_spectrum=False, filename=None):
        """Plots one or several components in the model.

        Parameters
        ----------
        components: {list of components, None}
            If list of components, plots all the components in the list.
            If None plots all the components in the model, which is the
            default.
        plot_combined: bool
            Plots the sum of the components, in addition to the individual
            components.
        filename: {string, None}
            If string, saves the plot to a file with name filename.
            If None raise a window with the plot and return the figure.

        Example:
        --------
        s = signals.Spectrum({'data':np.zeros(1000)})
        g1 = components.Gaussian()
        g2 = components.Gaussian()
        g3 = components.Gaussian()
        m = create_model(s)
        m.append(g1)
        m.append(g2)
        m.append(g3)
        m.plot_components() # To plot all components
        m.plot_components([g1]) # Plot just g1
        m.plot_components([g1,g2]) # Plot g1 and g2
        m.plot_components([g1,g2], filename="test.png") # Save to file

        """
        if not ((type(components) is list) or (components is None)):
            #Raise some relevant error
            print("Input must be in the form of a list of components:\
                    m.plot_components([component1, component2])")
            return

        axis = self.spectrum.axes_manager.signal_axes[0]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        component_signal_sum = np.zeros(axis.size) 
        if components is None:
            for component_ in self:
                component_signal = component_.function(axis.axis)
                component_spectrum = Spectrum({'data':component_signal})
                component_spectrum.axes_manager.signal_axes[0] = axis
                utils._make_mosaic_subplot(component_spectrum, ax)
                component_signal_sum += component_signal
        elif type(components) is list:
            for component_ in components:
                component_signal = component_.function(axis.axis)
                component_spectrum = Spectrum({'data':component_signal})
                component_spectrum.axes_manager.signal_axes[0] = axis
                utils._make_mosaic_subplot(component_spectrum, ax)
                component_signal_sum += component_signal
        
        if plot_combined:
            component_spectrum_sum = Spectrum({'data':component_signal_sum})
            component_spectrum_sum.axes_manager.signal_axes[0] = axis
            utils._make_mosaic_subplot(component_spectrum_sum, ax)
            
        if plot_spectrum:
            utils._make_mosaic_subplot(self.spectrum, ax)

        if filename is None:
            return(fig)
        else:
            fig.savefig(filename) 

    def generate_spectrum_from_components(self, components=None):
        """Generate a spectrum from one or several components in the model.

        Parameters
        ----------
        components: {list of components, None}
            If list of components, generates spectrum from all the 
            components in the list.
            If None plots all the components in the model, which is the
            default.

        Example:
        --------
        s = signals.Spectrum({'data':np.zeros(1000)})
        g1 = components.Gaussian()
        g2 = components.Gaussian()
        g3 = components.Gaussian()
        m = create_model(s)
        m.append(g1)
        m.append(g2)
        m.append(g3)
        m.generate_spectrum_from_components() # To plot all components
        m.generate_spectrum_from_components([g1,g2]) 
        m.generate_spectrum_from_components([g1])

        """
        if not ((type(components) is list) or (components is None)):
            #Raise some relevant error
            print("Input must be in the form of a list of components:\
                    m.plot_components([component1, component2])")
            return

        signal_axis = copy.deepcopy(self.spectrum.axes_manager.signal_axes[0])
        navigation_axis = copy.deepcopy(self.spectrum.axes_manager.navigation_axes[0])

        if components is None:
            for component_ in self:
                component_.active = True 
            self.generate_data_from_model()
            component_spectrum = Spectrum({'data':self.model_cube})
            component_spectrum.axes_manager.signal_axes[0] = signal_axis
            component_spectrum.axes_manager.navigation_axes[0] = navigation_axis
            return(component_spectrum)
        elif type(components) is list:
            for component_ in self:
                if component_ in components:
                    component_.active = True 
                else:
                    component_.active = False
            self.generate_data_from_model()
            component_spectrum = Spectrum({'data':self.model_cube})
            component_spectrum.axes_manager.signal_axes[0] = signal_axis
            component_spectrum.axes_manager.navigation_axes[0] = navigation_axis
            return(component_spectrum)

#ONLY WORKS FOR A SINGLE SPECTRUM
#        axis = self.spectrum.axes_manager.signal_axes[0]
#
#        component_signal_sum = np.zeros(axis.size) 
#        if components is None:
#            for component_ in self:
#                component_signal_sum += component_.function(axis.axis)
#            component_spectrum = Spectrum({'data':component_signal_sum})
#            component_spectrum.axes_manager.signal_axes[0] = axis
#            return(component_spectrum)
#        elif type(components) is list:
#            for component_ in components:
#                component_signal_sum += component_.function(axis.axis)
#            component_spectrum = Spectrum({'data':component_signal_sum})
#            component_spectrum.axes_manager.signal_axes[0] = axis
#            return(component_spectrum)

    def get_model_as_dict(self, filename=None):
        """Returns the spectrum, model and components in a dict.
        If filename is supplied, the dictionary will be saved as a
        numpy npz file.

        Parameters:
        -----------
        filename: string, None (default)
            If string will save the dictionary as a numpy npz file.

        Example:
        --------
        model_dict = m.get_model_as_dict()

        m.get_model_as_dict(filename="model_data.npz")
            
        """

        axis = self.axes_manager.signal_axes[0].axis 
        model_dict = {}
        component_list_dict = {}
        for component in self:
            component_dict = {}
            component_dict['name'] = component.__dict__['name']
            component_dict['type'] = component.__dict__['_id_name']
            parameter_list_dict = {}
            for parameter in component.__dict__['parameters']:
                parameter_dict = {}
                parameter_dict['name'] = parameter.__dict__['name'] 
                parameter_dict['map'] = parameter.__dict__['map']
                parameter_list_dict[parameter.__dict__['name']] = parameter_dict
            component_dict['parameters'] = parameter_list_dict
            component_dict['component_spectrum'] = self.generate_spectrum_from_components([component,]).data

            component_list_dict[component_dict['name'] + component_dict['type']] = component_dict

        model_dict['components'] = component_list_dict
        model_dict['spectrum'] = self.spectrum.data
        model_dict['model_spectrum'] = self.generate_spectrum_from_components().data
        model_dict['navigational_axis'] = self.axes_manager.navigation_axes[0].axis
        model_dict['navigational_axis_units'] = self.axes_manager.navigation_axes[0].units
        model_dict['signal_axis'] = self.axes_manager.signal_axes[0].axis
        model_dict['signal_axis_units'] = self.axes_manager.signal_axes[0].units


        if filename is None:
            return(model_dict)
        else:
            np.savez(filename, model_dict=model_dict)

    def plot_model_report(self, title='', filename=None, figsize=(10,10)):
        figure, subplots = plt.subplots(1, 4, figsize=figsize)
        
        #Todo: 
        #signal_axis not working for cascade_model
        #add parameters to plot

        cascade_spectrum = subplots[0]
        utils._make_cascade_subplot(self.spectrum, cascade_spectrum)

        cascade_model = subplots[1]
        model_as_spectrum = self.generate_spectrum_from_components()
        utils._make_cascade_subplot(model_as_spectrum, cascade_model)

        error_subplot = subplots[3]
        #Calculate the error
        spectrum_difference = np.abs(model_as_spectrum.data-self.spectrum.data)**2
        total_intensity = np.sum(self.spectrum.data, axis=1)
        error_array = np.sqrt(np.sum(spectrum_difference, axis=1))/total_intensity
        utils._plot_cascade_parameter(error_array, error_subplot) 

        figure.suptitle(title)
        if filename == None:
            return(figure)
        else:
            figure.savefig(filename)
