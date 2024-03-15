*Cellarium ML: distributed single-cell data analysis.*

---------

Installation
------------

To install from the pip::

   $ pip install cellarium-ml

To install the developer version from the source::

   $ git clone https://github.com/cellarium-ai/cellarium-ml.git
   $ cd cellarium-ml
   $ make install               # runs pip install -e .[dev]

For developers
--------------

To run the tests::

   $ make test                  # runs single-device tests
   $ TEST_DEVICES=2 make test   # runs multi-device tests

To automatically format the code::

   $ make format               # runs ruff and black formatters

To run the linters::

   $ make lint                  # runs ruff and black checks

To build the documentation::

   $ make docs                  # builds the documentation at docs/build/html


