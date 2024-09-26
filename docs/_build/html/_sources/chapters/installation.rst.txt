Installation
------------

System requires
***************
The software package can be installed and runned on Linux, Windows, and MacOS 

Dependency of Python and Python packages: 


.. code-block:: bash

    python == 3.9
    numpy == 1.26.1
    scipy == 1.11.4
    torch == 1.13.1
    tqdm == 4.66.1

.. note::
    1. Versions that has been previously tested on are also listed below, other versions should work the sameersions that has been previously tested on are listed above, other versions should work the same.

    2. The required python packages with the latest versions will be automatically installed if these python packages are not already present in your local Python environment.


Installation for source
***********************
1. Download and install the latest `Anaconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_ distribution:

.. code-block:: bash

    wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
    ./Anaconda3-2024.06-1-Linux-x86_64.sh


2. Create a new ``conda`` virtual environment and install the ts-dart source code locally:

.. code-block:: bash

    conda create -n ts-dart python=3.9
    conda activate ts-dart
    git clone https://github.com/xuhuihuang/ts-dart.git
    python -m pip install ./ts-dart
