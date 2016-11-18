PolyOrder
=========

PolyOrder is a C++ library and a robust framework for developing numerical polymer self-consistent field theory (SCFT) software.

Quickstart
----------

1. Install
^^^^^^^^^^

To make ``liborder.a``::

    $ make lib

You need to move the ``liborder.a`` from the ``build/lib`` directory to a directory known by your compiler. A gcc 4.5+ is recommended for building this library.

**Requirements**

* FFTW 3.3+
* Armadillo 4.3+
* blitz++ (version contains ``tinyvec-et.h``)
* ``CMatFile.h`` and MatIO (``-lmatio``)
* ``Simpleini.h`` + ``ConvertUTF.h`` + ``ConvertUTF.c``

Additional library for charged polymers

* ``libndarray.a`` (``-lndarray``, for ``CNDArray``)

*NOTE: For Mac OS x86_64, add ``-m64`` to ``CXXFLAGS`` in the Makefile*

2. Usage
^^^^^^^^

A typical use of PolyOrder is to write a ``main`` function which completes following tasks:

* Select a polymer model and initialize it with an appropriate configuration file.
* Use the created model instance to initialize the SCFT driver.
* Run the driver instance.
* Optional: you can record the run time.

Examples can be found in the ``scft`` directory.

To build your application, just compile and link your code with the PolyOrder library. You can also modify the Makefile in the root directory and build your application by

::

    $ make main

Ask for Help
------------

Let me know what you think and wish. I can be reached through `email <mailto:liuyxpp@gmail.com>`_ or `other ways <http://ngpy.org/about>`_.

Links
-----

* `Yi-Xin Liu's personal website <http://ngpy.org>`_
* Source code hosted by `GitHub <https://github.com/liuyxpp/polyorder>`_.
