# Modern Applications of Numerical Linear Algebra Methods

This is the collection of course materials for the GIAN Course
_Modern Applications of Numerical Linear Algebra Methods_.

[Browse the notebooks](http://nbviewer.jupyter.org/url/github.com/ivanslapnicar/GIAN-Applied-NLA-Course/tree/master/src/)

The course was held from _June 26, 2016_ to _July 5, 2016_ at IIT Indore. The notebooks are currently being used in the course "Numerical Linear Algebra" at the University of Split.

Here are some links:
* [Global Initiative of Academic Networks](http://www.gian.iitkgp.ac.in/)
* [List of Approved Courses](http://www.gian.iitkgp.ac.in/ccourses/approvecourses2)
* [Course Description](BR1458109755Final_Modern_Applications_of_Numerical_Linear_Algebra_Methods.PDF)
* [Indian Institute of Technology, Indore](http://www.iiti.ac.in/)


The course consists of three modules:

* Module A - Short Julia Course
* Module B - Eigenvalue and Singular Value Decompositions
* Module C - Applications

The course is presented through twenty  one-hour lectures and ten two-hour tutorials.

Course materials are using [Julia](http://julialang.org/) programming language
and are presented as [Jupyter](http://jupyter.org/) notebooks.
The naming scheme of the notebooks is the following:

* lectures start with `L`,
* tutorial assignments start with `T`, and
* solutions to tutorial assignments start with`S`.

PDF files for the three modules are also provided. Meanwhile, the notebooks are being updated and ported to the current version of Julia.

To understand the author's concepts behind the creation of the course,
read the [Manifest](src/Manifest.md).

The notebooks can be used in three ways:

1. Browsing the notebooks with
[Jupyter notebook viewer](http://nbviewer.jupyter.org/) ->
[browse the notebooks](http://nbviewer.jupyter.org/url/github.com/ivanslapnicar/GIAN-Applied-NLA-Course/tree/master/src/)

2. Cloning the notebooks to your computer with the command (in Linux)

    `git clone https://github.com/ivanslapnicar/GIAN-Applied-NLA-Course.git`

    For Windows, you can install the [GitHub Desktop](https://desktop.github.com/)
    and use it to clone the repository.

    The notebooks are now located in the directory `GIAN-Applied-NLA-Course/src/` and can
be used interactively (you need to install  [Julia](http://julialang.org/) and
[Jupyter](http://jupyter.org/) as described in the notebook
[L00 Installing and Running Julia](http://nbviewer.jupyter.org/url/github.com/ivanslapnicar/GIAN-Applied-NLA-Course/tree/master/src/Module A - Short Julia Course/L00 Installing and Running Julia.ipynb)).
PDF files used in original course are in the directory `src/pdf`.
PDF files of the updated notebooks are in the directory `src/pdf/new`.

3. Executing the notebooks on [JuliaBox](https://juliabox.com/):

  * go to https://juliabox.com and sign in
  * go to __Git__
  * paste the address `https://github.com/ivanslapnicar/GIAN-Applied-NLA-Course.git` into
__Git Clone URL__ box
  * check that the __Branch__ is set to _master_
  * press _+_ to add the repository
  * close the __Git__ window

  The directory `GIAN-Applied-NLA-Course` is now listed. The notebooks are
  located in the directory `src/`.
