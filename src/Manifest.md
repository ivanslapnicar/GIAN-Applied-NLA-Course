# Manifest - Teaching - Julia - Research

This is the author's Teaching Manifest and Call for Collaboration.

---

## Teaching

### Starting point

Currently there is documented and elaborated need for improving
mathematical education, in particular in STEM areas.  The key
observation is that the pressure on math education, and STEM education
in general, is constantly mounting, in particular in the cuntries with
strong need for STEM workforce. This is, for example, made clear in the
* [PISA](http://www.oecd.org/pisa/) surveys and
* industry requests

The basic reasons for learning math can be summarized as
* an ability to perform tech jobs
* an ability to have high quality everyday living
* the development of logic(al) thinking for other areas

A complete real-world math (and more general, science and engineering)
knowledge would be comprised of four components:
1. __Defining__ of the real-world problem
2. __Modeling__ and translation of the problem into math
3. __Computating__ - solving the math problem
4. __Interpreting__ the solution - application to the real-world

Currently, there is obvious disproportion of the time spent on each of
these steps. Although there are lively discussion about needs and means
to change the education, it is clear that pupils and students at lower
levels spend most of the time (in the course of many years) on item 3.,
while other items are neglected. It can be argued that for many instances
of computating, computers (computer algebra systems, programing
languages, etc.) can be used as very successful tools, freeing sufficient
study time to access other items better. Even though such efforts are
not welcome for the entire population, it is non-arguable that certain
steps in this direction need to be taken.

### Notebooks and Cloud

There is considerable effort to do so. Notebook principle introduced
in [Mathematica](http://www.mathematica.com) or
[Jupyter](https://jupyter.org) and computing in the cloud like
[WolframAlpha](http://www.wolframalpha.com/) or
[JuliaBox](http://juliabox.org) hosted by
[Amazon Web Services](http://aws.amazon.com/), or
other solutions, enable teachers and others to quickly create efficient,
illustrative and interactive materials which have a great potential to
enhance learning and teaching experience, speed up the acquisition of
knowledge and respond well to the existing needs.

### Crowdsourcing

The creation of (unofficial) math instruction materials is being
intensively rowdsourced. Good examples are Wolfram's Mathematica
Notebooks, [Computable Document
Format](http://www.wolfram.com/cdf/?source=nav) and Alpha, and geogebra's
[apps](http://web.geogebra.org/app/), written by many


The next step is to combine (and formalize) crowdsourcing with the
state-of-the art collaborative (open source) code development systems
like [github](https://github.com/), and include all
stakeholders (other teachers, users, industry experts and students) in
the process of course creation, giving them opportunity to _directly_
contribute to the course design.

> Please, be social and join!

### MOOCs

Massive open-online courses, such as those produced and managed on
[Coursera](http://coursera.org), [edX](http://edx.org), or
[MITx](https://www.edx.org/school/mitx/allcourses), are an intensive
development in education, made possible by the current development of
internet technologies.

Impressive list of schools and partners involved in creating materials
for both platforms, serves both, as the quality assurance and a
persuasion factor for learners.

The courses, although produced by the world-class faculty, currently
ensure neither broad enough base of stakeholders involved in the course
creation, nor is the content sufficiently interactive.

### Open Source

There are many high-quality open-source endeavors, highly usable in math
and STEM education:
* [Octave](http://octave.org),
* [Python](http://python.org),
* [Jupyter](http://jupyter.org),
* [Julia](http://julialang.org),
* [edX](http://edx.org), ...

to name a few, many being developed on github, see:

* https://github.com/jupyter
* https://github.com/JuliaLang
* https://github.com/edx

making open source preferred choice for the development of math and
STEM courses.

Particularly, the recent project [Jupyter](http://jupyter.org/)
develops notebook technology which is:
* language-agnostic version,
* has a system for assigning and grading notebooks,
[nbgrader](https://github.com/jupyter/nbgrader), and
* set of tools for deployment of multi-user servers for Jupyter
notebooks, including usage of cloud technologies.

### Development

We advocate development of courses for STEM education, with the following
features:

* courses designed by using Julia notebooks, thus having simple and
efficient interactivity,
* course material developed on Github, thus being open source,
* collaborative course creation with course material development
being crowdsourced whenever possible, that is
"Share the hard part" v.s. "You have to do the hard part for
yourself" (Stevie Ray Vaughn)
* courses deployed as MOOCs on edX os similar platform by
teachers acting as direction setting moderators and
persuading and quality-assurance factor,
*  courses being (almost) automatically graded using `nbgrader` or
`edX` technology.


## Julia

[Julia](http://julialang.org) is a high level dynamic language which
combines ease of use of dynamic high-level languages with the speed of
static languages as FORTRAN or C. Besides being a suitable language for
modern introductory computer science course, it is ideal to teach
STEM courses.

### Features

* fast
* open
* profiling
* types
* mutliple dispatch (polymorphism)
* operator and function overloading
* multidimensional arrays
* packages
* writing your packages on [github](https://github.com)
* many plotting environments
* interfaces to other languages
* interfaces to software packages (like [LAPACK](http://www.netlib.org/lapack))
* can be used without installing it at [JuliaBox](http://juliabox.org)
* [Jupyter](http://jupyter.org) notebooks
* symbolic computation with [sympy](http://sympy.org/en/index.html)
* interactivity with [@manipulate](https://github.com/JuliaLang/Interact.jl)
* web accesss
* handling big data with ([HDF5](http://www.hdfgroup.org/HDF5/))
* great to use in [teaching](http://julialang.org/teaching/) and research


### Culture

Julia is a great open-source project under constant development.

Culture of Julia is complete openness.
If something does not work, or does not work as expected, or if you have
questions or want interaction with other users or developers, use any of
available means listed on [Julia
Community](http://julialang.org/community/) page:

* [blog](http://julialang.org/blog/),
* mailing lists like [julia-users](https://groups.google.com/forum/?fromgroups=#!forum/julia-users) and  [julia-dev](https://groups.google.com/forum/?fromgroups=#!forum/julia-dev),
* on GitHub
    * [issues](https://github.com/JuliaLang/julia/issues) (bugs and their
    fixes, new features, and proposed changes), and
    * [commits](https://github.com/JuliaLang/julia/commits) (latest
    changes to the development version).


## Research

The notebook technology is also a great open source tool to conduct,
record  and exchange everyday research.

