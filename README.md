# SpaceyPy
Some home-grown Orbital Dynamics and Mission Design tools written in Python. This is predominantly a learning expercise, definitely not supposed to be a competitive production ready library. Just a couple utils and fun demos of spacey stuff. 

Utilities built with a mix of existing (and awesome) `poliastro` and `astropy` libraries, as well as some home grown methods.

## Setup

### Python
You just need a valid python 3.5 environment to get rolling. Either install dependencies yourself in existing environment, or use Anaconda. 

Anaconda's Python distribution is recomended for getting started. Easiest way to get started is to install and create a new Python 3.5 environment from the included (not minimal) `py-environment.yml` file. 
http://conda.pydata.org/docs/using/envs.html

```
$ conda env create -f py-environment.yml
```
Otherwise, if you don't have Anaconda, a working Python 3.5+ environment and a few ancilary modules is all you need. 

## Example
Run example notebook for some samples. Note the use of `astropy.units` for all calculations. 
![Simple Porkchop](http://i.imgur.com/PUKyoQr.png)

![Simple Interplanetary Trajectory Plotter](http://i.imgur.com/MSlBB8X.png)

# Todos
Endless! 

* Tests with pytest
* Build a Propegator
* Low thrust approximation with `poliastro.maneuver` objects
* * Higher fidelity low thrust design class and propegator
* Documentation on ancilary functions
* Fix label positions in interplanetary trajectory plot
* Figure out how to do flybys (lol like it just deserves one bullet)
* More Spice integration
