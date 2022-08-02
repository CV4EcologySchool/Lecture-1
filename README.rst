
================================================
Lecture 1: Dataset Prototyping and Visualization
================================================

.. contents:: Quick Links
    :backlinks: none

.. sectnum::

Introduction
------------

.. image:: https://github.com/CV4EcologySchool/Lecture-1/raw/main/intro.jpg
    :alt: "Lecture 1: Dataset Prototyping and Visualization"

This repository holds the lecture materials for the `Computer Vision for Ecology workshop at CalTech <https://cv4ecology.caltech.edu>`_.  The goal of this lecture is to describe which qualities are idea for prototype ML datasets and a review of PyTorch's DataLoaders and Tensors.  Lecture 1 also reviews the overall lecture structure for the three week course, review the milestone due tomorrow (Week 1, Day 2), review the tools and technologies and terms that are common for ML applications.

Lecture Slides
--------------------------

The associated slides for this lecture can be viewed by clicking on the image below.

.. image:: https://github.com/CV4EcologySchool/Lecture-1/raw/main/slides.jpg
    :target: https://docs.google.com/presentation/d/1Bm9ZvQC6Y1SW_xAHHbMvhsRRb87tgzIimM0YKEXEA6w/edit?usp=sharing
    :alt: View the Google Slide presentation

Overiew of Tools
----------------

.. image:: https://github.com/CV4EcologySchool/Lecture-1/raw/main/tools.jpg
    :alt: Technologies, Tools, and Terms

Week 1, Day 2 Milestone Checklist
---------------------------------

.. image:: https://github.com/CV4EcologySchool/Lecture-1/raw/main/checklist.jpg
    :alt: Milestone Checklist

How to Install
--------------

You need to first install Anaconda on your machine.  Below are the instructions on how to install Anaconda on an Apple macOS machine, but it is possible to install on a Windows and Linux machine as well.  Consult the `official Anaconda page <https://www.anaconda.com>`_ to download and install on other systems.  For Windows computers, it is highly recommended that you intall the `Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/install>`_.

.. code:: bash

   # Install Homebrew
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

   # Install Anaconda and expose conda to the terminal
   brew install anaconda
   export PATH="/opt/homebrew/anaconda3/bin:$PATH"
   conda init zsh
   conda update conda

Once Anaconda is installed, you will need an environment and the following packages installed

.. code:: bash
   
   # Create Environment
   conda create --name cv4e
   conda activate cv4e

   # Install Python dependencies
   conda install numpy matplotlib ipython jupyter opencv rich click
   conda install pytorch torchvision -c pytorch-nightly

   # Optional
   conda install pip pre-commit 
   pip install brunette

How to Run
----------

The lecture materials will run as a single executable.  The MNIST dataset must be downloaded from the internet for this script to run correctly, so Internet access is required at first to download the files once.  It is recommended to use `ipython` and to copy sections of code into and inspecting the 

.. code:: bash
   
   # Run with Python
   python lecture.py

   # Run with iPython
   ipython lecture.py

   # Run as an executable
   ./lecture.py

How to use Jupyter Notebook
---------------------------

.. image:: https://github.com/CV4EcologySchool/Lecture-1/raw/main/notebook.jpg
    :alt: Jupyter Notebook for Lecture 1

The lecture may also be run from within a Jupyter Notebook, which is an online server that gives you the ability to run Python code in blocks and store in-line results and notes.

.. code:: bash
   
   # Run the Jupyter Notebook server
   jupyter notebook

   # Navigate to http://localhost:8888/notebooks/notebook.ipynb in your browser (or whatever URL your notebook server is listening for)

Logging
-------

The script uses Python's built-in logging functionality called `logging`.  All print functions are replaced with `log.info` within this script, which sends the output to two places: 1) the terminal window, 2) the file `lecture_1.log`.  Get into the habit of writing text logs and keeping date-specific versions for comparison and debugging.

Code Formatting (Optional)
--------------------------

It's recommended that you use ``pre-commit`` to ensure linting procedures are run
on any code you write. (See also `pre-commit.com <https://pre-commit.com/>`_)

Reference `pre-commit's installation instructions <https://pre-commit.com/#install>`_ for software installation on your OS/platform. After you have the software installed, run ``pre-commit install`` on the command line. Now every time you commit to this project's code base the linter procedures will automatically run over the changed files.  To run pre-commit on files preemtively from the command line use:

.. code:: bash

    git add .
    pre-commit run

    # or

    pre-commit run --all-files

The code base has been formatted by Brunette, which is a fork and more configurable version of Black (https://black.readthedocs.io/en/stable/).  Furthermore, try to conform to PEP8.  You should set up your preferred editor to use flake8 as its Python linter, but pre-commit will ensure compliance before a git commit is completed.  This will use the flake8 configuration within ``setup.cfg``, which ignores several errors and stylistic considerations.  See the ``setup.cfg`` file for a full and accurate listing of stylistic codes to ignore.

See Also
--------

- https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
- https://pytorch.org/vision/stable/datasets.html
- https://pytorch.org/audio/stable/datasets.html
- https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
- https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#sphx-glr-auto-examples-plot-visualization-utils-py
