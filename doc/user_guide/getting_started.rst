Getting started
***************

First steps with hyperspy
=========================

Starting hyperspy
-----------------

Hyperspy is a Python library to analyze multidimensional. Hyperspy does not
have a GUI. Instead, the most common way of running Hyperspy is interactively
using the wonderful interactive computing package `IPython
<http://ipython.org>`_. In this section we describe the easiest ways to start
hyperspy in the different operating systems.


Starting hyperspy from the terminal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In all operating systems (OS) you can start Hyperspy
by opening a system terminal and typing hyperspy:

.. code-block:: bash

    $ hyperspy


If Hyperspy is correctly installed it should welcome you with a message similar
to:

.. code-block:: ipython
    
    H y p e r s p y
    Version 0.7
    
    http://www.hyperspy.org	
	

If IPython 0.11 or newer and the Qt libraries are installed in your system it
is also possible to run Hyperspy in `IPython's QtConsole
<http://ipython.org/ipython-doc/stable/interactive/qtconsole.html>` by
executing `hyperspy qtconsole` in a terminal:

.. code-block:: bash

    $ hyperspy qtconsole

If IPython 0.12 or newer is installed in your system it is also possible to run
Hyperspy in `IPython's HTML notebook
<http://ipython.org/ipython-doc/stable/interactive/htmlnotebook.html>`_ that
runs inside your browser. The Notebook is probably **the most agreeable way** to work
with Hyperspy interactively. You can start it from a terminal as follows

.. code-block:: bash

    $ hyperspy notebook 

There are multiple options available when starting from the terminal. To print
these options add the `-h` flag:

.. code-block:: bash

    $ hyperspy -h
    usage: hyperspy [-h] [-v] [--toolkit {qt4,gtk,wx,tk,None}]
                    [--pylab_inline] [--overwrite_profile]
                    [--ipython_args [IPYTHON_ARGS [IPYTHON_ARGS ...]]]
                    [{terminal,console,qtconsole,notebook}]

    Hyperspectral data analysis toolbox
 
    positional arguments:
      {terminal,console,qtconsole,notebook}
                            Selects the IPython environment in which to start
                            Hyperspy. The default is terminal
 
    optional arguments:
      -h, --help            show this help message and exit
      -v, --version         show program's version number and exit
      --toolkit {qt4,gtk,wx,tk,None}
                            Pre-load matplotlib and traitsui for interactive
                            use,selecting a particular matplotlib backend and loop
                            integration.When using gtk and tk toolkits the user
                            interface elements are not available. None is suitable
                            to run headless.
      --pylab_inline        If True the figure are displayed inline. This option
                            only has effect when using the qtconsole or notebook
      --overwrite_profile   Overwrite the Ipython profile with the default one.
      --ipython_args [IPYTHON_ARGS [IPYTHON_ARGS ...]]
                            Arguments to be passed to IPython. This option must be
                            the last one.Look at the IPython documentation for
                            available options.
 
Starting hyperspy from the context menu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This option is only available for Windows and in Linux when using Gnome.

Windows
"""""""

In Windows it is possible to start Hyperspy from :menuselection:`Start Menu -->
Programs --> Hyperspy`.

Alternatively, one can start Hyperspy in any folder by pressing the :kbd:`right
mouse button` or on a yellow folder icon or (in some cases) on the empty area
of a folder, and choosing :menuselection:`Hyperspy qtconsole here` or
:menuselection:`Hyperspy notebook here` from the context menu.


.. figure::  images/windows_hyperspy_here.png
   :align:   center
   :width:   500    

   Starting hyperspy using the Windows context menu.
   

Linux
"""""

If you are using GNOME in Linux, you can open a terminal in a folder by
choosing :menuselection:`open terminal` in the file menu if
:program:`nautilus-open-terminal` is installed in your system.

Altenatively (and more conviently), if you are using Gnome place `this
<https://github.com/downloads/hyperspy/hyperspy/Hyperspy%20QtConsole%20here.sh>`_
and `this
<https://github.com/downloads/hyperspy/hyperspy/Hyperspy%20Notebook%20here.sh>`_
in the :file:`/.gnome2/nautilus-scripts` folder in your home directory (create
it if it does not exists) and make them executable to get the
:menuselection:`Scripts --> Hyperspy QtConsole Here` and
:menuselection:`Scripts --> Hyperspy Notebook Here` entries in the context
menu. 


.. figure::  images/hyperspy_here_gnome.png
   :align:   center
   :width:   500    

   Starting hyperspy using the Gnome nautilus context menu.

Using Hyperspy as a library
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When starting hyperspy by using the file browser context menu or by running the
`hyperspy` script in the terminal, the `hyperspy` script simply starts and
configure IPython appropiately and import the contents of the :py:mod:`~.hspy`
module in the user namespace. Of course, It is possible to use Hyperspy as a
library by simply importing its modules. The recommended way to do it is
importing the hspy module as follows:

.. code-block:: python

    import hyperspy.hspy as hp

Choosing a toolkit
------------------
.. versionadded:: 0.7

Hyperspy fully supports the Qt toolkit in addition to the WX toolkit. GTK and
TK are also supported but the user interface elements are not available. The
default toolkit is Qt4/PySide because currently is the one that works best in
most platforms.

The toolkit can be specified on start using the `--toolkit` flag. Also, the
default value can be configured in :ref:`preferences <configuring-hyperspy-label>`.

Inline plotting
---------------
.. versionadded:: 0.7

Hyperspy can be started using the IPython inline plotting mode by passing the
`--pylab_inline` flag when starting hyperspy. This option only has effect when
running in the *IPython QtConsole* or the *IPython Notebook*, e.g.

.. code-block:: bash
   $ hyperspy qtconsole --pylab_inline

The default value can be configured in :ref:`preferences
<configuring-hyperspy-label>`.

.. _headless-label:

Using Hyperpsy in a headless system
-----------------------------------
.. versionadded:: 0.7

To run Hyperpsy in a headless system select "None" as the toolkit either in 
:ref:`preferences <configuring-hyperspy-label>` or when starting from a
terminal, e.g.:


.. code-block:: bash

    $ hyperspy --toolkit None 


Getting help
------------

The documentation (docstring in Python jargon) can be accessed by adding a
question mark to the name of a function. e.g.:

.. code-block:: python
    
    >>> load?

This syntax is a shortcut to the standard way one of displaying the help
associated to a given functions (docstring in Python jargon) and it is one of
the many features of `IPython <http://ipython.scipy.org/moin/>`_ , which is the
interactive python shell that Hyperspy uses under the hood.

Please note that the documentation of the code is a work in progress, so not
all the objects are documented yet.

Up-to-date documentation is always available in `the Hyperspy website.
<http://hyperspy.org/documentation.html>`_


Autocompletion
--------------

Another useful `IPython <http://ipython.scipy.org/moin/>`_ feature is the
autocompletion of commands and filenames using the tab and arrow keys. It is
highly recommended to read the `Ipython documentation
<http://ipython.scipy.org/moin/Documentation>`_ (specially their `Getting
started <http://ipython.org/ipython-doc/stable/interactive/tutorial.html>`_
section) for many more useful features that will boost your efficiency when
working with Hyperspy/Python interactively.


Loading data
------------

Once hyperspy is running, to load from a supported file format (see
:ref:`supported-formats`) simply type:

.. code-block:: python

    >>> s = load("filename")

.. HINT::

   The load function returns an object that contains data read from the file.
   We assign this object to the variable ``s`` but you can choose any (valid)
   variable name you like. for the filename, don't forget to include the
   quotation marks and the file extension.
   
If no argument is passed to the load function, a window will be raised that
allows to select a single file through your OS file manager, e.g.:

.. code-block:: python

    >>> # This raises the load user interface
    >>> s = load()

It is also possible to load multiple files at once or even stack multiple
files. For more details read :ref:`loading_files`


.. _saving:

Saving Files
------------

The data can be saved to several file formats.  The format is specified by
the extension of the filename.

.. code-block:: python

    >>> # load the data
    >>> d = load("example.tif")
    >>> # save the data as a tiff
    >>> d.save("example_processed.tif")
    >>> # save the data as a png
    >>> d.save("example_processed.png")
    >>> # save the data as an hdf5 file
    >>> d.save("example_processed.hdf5")

Some file formats are much better at maintaining the information about
how you processed your data.  The preferred format in Hyperspy is hdf5,
the hierarchical data format.  This format keeps the most information
possible.

There are optional flags that may be passed to the save function. See
:ref:`saving_files` for more details.


.. _configuring-hyperspy-label:

Configuring hyperspy
--------------------

The behaviour of Hyperspy can be customised using the
:py:class:`~.defaults_parser.Preferences` class. The easiest way to do it is by
calling the :meth:`gui` method:

.. code-block:: python

    >>> preferences.gui()
    
This command should raise the Preferences user interface:

.. _preferences_image:

.. figure::  images/preferences.png
   :align:   center

   Preferences user interface

.. _getting-help-label:



Data visualisation
==================

The object returned by :py:func:`~.io.load` is a :py:class:`~.signal.Signal`
and has a :py:meth:`~.signal.Signal.plot` method which plots the data and
allows navigation.

.. code-block:: python
    
    >>> s = load('YourDataFilenameHere')
    >>> s.plot()

if the object is single spectrum or an image one window will appear when
calling the plot method.

If the object is a 1D or 2D spectrum-image (i.e. with 2 or 3 dimensions when
including energy) two figures will appear, one containing a plot of the
spectrum at the current coordinates and the other an image of the data summed
over its spectral dimension if 2D or an image with the spectral dimension in
the x-axis if 1D:

.. _2d_SI:

.. figure::  images/2D_SI.png
   :align:   center
   :width:   500

   Visualisation of a 2D spectrum image
   
.. _1d_SI:

.. figure::  images/1D_SI.png
   :align:   center
   :width:   500

   Visualisation of a 1D spectrum image
   
Equivalently, if the object is a 1D or 2D image stack two figures will appear, 
one containing a plot of the image at the current coordinates and the other
a spectrum or an image obtained by summing over the image dimensions:
   
.. _1D_image_stack.png:

.. figure::  images/1D_image_stack.png
   :align:   center
   :width:   500    

   Visualisation of a 1D image stack
   
.. _2D_image_stack.png:

.. figure::  images/2D_image_stack.png
   :align:   center
   :width:   500
   
   Visualisation of a 2D image stack

To change the current coordinates, click on the pointer (which will be a line
or a square depending on the dimensions of the data) and drag it around. It is
also possible to move the pointer by using the numpad arrows **when numlock is
on and the spectrum or navigator figure is selected**.When using the numpad
arrows the PageUp and PageDown keys change the size of the step.

An extra cursor can be added by pressing the ``e`` key. Pressing ``e`` once
more will disable the extra cursor:

.. _second_pointer.png:

.. figure::  images/second_pointer.png
   :align:   center
   :width:   500

   Visualisation of a 2D spectrum image using two pointers.

When exploring a 2D hyperspectral object of high spatial resolution the default
size of the rectangular cursors can be too small to be dragged or even seen. It
is possible to change the size of the cursors by pressing the ``+`` and ``-``
keys  **when the navigator window is selected**.

The same keys can be used to explore an image stack.



=========   =============================
key         function    
=========   =============================
e           Switch second pointer on/off
Arrows      Change coordinates  
PageUp      Increase step size
PageDown    Decrease step size
``+``           Increase pointer size
``-``           Decrease pointer size
``h``       Launch the contrast adjustment tool (only for Image)
=========   =============================

To close all the figures run the following command:

.. code-block:: python

    close('all')

.. NOTE::

    This is a `matplotlib <http://matplotlib.sourceforge.net/>`_ command.
    Matplotlib is the library that hyperspy uses to produce the plots. You can
    learn how to pan/zoom and more  `in the matplotlib documentation
    <http://matplotlib.sourceforge.net/users/navigation_toolbar.html>`_



