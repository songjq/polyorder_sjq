Format of Configuration File
============================

Grid
----

Required parameters list
~~~~~~~~~~~~~~~~~~~~~~~~

* ``dimension``
* ``Lx``
* ``Ly``
* ``Lz``
* ``grid_type_x``
* ``grid_type_y``
* ``grid_type_z``
* ``gridInitType``

Optional parameters list
~~~~~~~~~~~~~~~~~~~~~~~~

* ``confine_geometry``
* ``BC_coefficients_left``
* ``BC_coefficients_right``
* ``random_seed``
* ``field_data``

``confine_geometry``: optional
Allowed keywords (case-insensitive):
    [line, slit, slab, cube, disk, cylinder, sphere]
others mean no confinement.
* line: 1D line geometry, confined by two end points.
* slit: 2D rectangular geometry, confined by two lines.
* slab: 3D cuboid geometry, confined by two planes.
* cube: according to Grid.dimension, 1D => line, 2D => slit, 3D => slab.
* disk: 2D circular geometry, confined by circumference.
* cylinder: 3D cylindrical geometry, confined by cylindrical surface.
* sphere: 3D spherical geometry, confined by spherical surface.

Notes
-----

* Comments can be added as a "#"-leading-line.
* Key and value should be comparted by a equal sign, and all blank character
  between keys and values will be neglected automatically.
* The trailing whitespaces may cause serious problem, they should be removed carefully. (Note added by Yi-Xin Liu)
* Support section name. Section name is enclosed by square bracket.
* No difference will be caused by changing the suquences of the parameters.
* The version of this file (param.ini) should be the same as
  the version of the script paramx.
* BC vector is [a,b,c] where
               a du/dx + b u = c
* The min tol_cell = 3.0e-8.

Changelog
---------

Version 10.0
~~~~~~~~~~~~

* Add Version, Algorithm_MDE, Algorithm_SCFT, Algorithm_Cell_Optimization, Algorithm_Contour_Integration sections.
* Remove SCFT.is_batch_cell
* Move some keys to new sections
* Break SCFT section into IO and Algorithm_SCFT sections.
* Use tests/test_config_v10 to test whether a configuration file is consistent internally.

Version 9.0 and before
~~~~~~~~~~~~~~~~~~~~~~

* 2014.06.12 Compatible with current version of Polyorder
* 2013.07.31 Compatible with scftpy/bulk.
* 2013.06.24 Add q_file.
* 2012.11.08 Add BC support.
* 2012.10.11 Currently, only supports one type of chain.
* 2012.10.10 Break compatability with previous version. Only for scftpy use.
             Move Ms from [Model] to [Grid].
* 2012.4.24 Add dielectric_constant, charge_distribution to [Algorithm].
            Delete isAnnealed from [Model].
* 2012.4.22 Add [Algorithm] section
* 2012.4.20 Add fft2mg_mode and fft2mg_interp
* 2012.4.18 Add gensym.py support

ToDo List
---------

* Replace .ini format with JSON format. JSON format is more robust and common.
