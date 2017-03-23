#! /usr/local/bin/python
# -*- coding: utf-8 -*-
__author__     = 'Zach Dischner'
__copyright__  = "CYGNSS - Southwest Research Institute"
__credits__    = ["NA"]
__license__    = "NA"
__version__    = "0.0.0"
__maintainer__ = "Zach Dischner"
__email__      = "dischnerz@boulder.swri.edu"
__status__     = "Dev"
__doc__        ="""
File name: astro.py
Authors:  
Created:  Aug/04/2016
Modified: Sept/05/2016

Description:
    Holds some basic orbit dynamics computation and visualization utilities. 

Makes heavy use of the badass libraries already in use:
    astropy: http://www.astropy.org
    poliastro: http://poliastro.readthedocs.io/en/latest/

References: 
    Schaub/Junkins Analytical Mechanics of Space Systems

TODO:
    Auto tests
"""

##############################################################################
#                                   Imports
#----------*----------*----------*----------*----------*----------*----------*
import sys
import warnings
import os
import numpy as np
import pandas as pd
from numba import jit, njit, float64, autojit
from numpy.linalg import norm
from scipy.optimize import bisect,newton
from astropy import units as u
from astropy import time
from astropy.units.quantity import Quantity
from poliastro import ephem
from poliastro.bodies import Sun, Earth
from poliastro.twobody import Orbit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from orbitpy import utils
from orbitpy.stumpff import get_C, get_S, get_Cprime, get_Sprime

import spiceypy

##############################################################################
#                             Module-wide Variables
#----------*----------*----------*----------*----------*----------*----------*
## Directories
_here = os.path.dirname(os.path.realpath(__file__))

#--constants
j2Earth = 0.00108262668 # unitless
rEarth  = Earth.R
gmEarth = Earth.k



##############################################################################
#                             Helper Functions
#----------*----------*----------*----------*----------*----------*----------*


@jit(float64(float64,float64,float64,float64,float64))
def get_vars(z, A, r1_norm, r2_norm, mu):
    C = get_C(z)
    S = get_S(z)
    y = r1_norm + r2_norm - A*((1-z*S)/C**0.5)
    if A > 0 and y < 0:
        # Taken from: https://github.com/poliastro/poliastro/blob/master/src/poliastro/iod/vallado.py
            # Readjust z until y > 0.0
            # Translated directly from Vallado
        while y < 0.0:
            z = (0.8 * (1.0 / get_S(z)) * (1.0 - (r1_norm + r2_norm) * get_C(z)**0.5) / A)
            y = r1_norm + r2_norm - A*((1-z*S)/C**0.5)

    x       = (y/C)**0.5
    tof_new = (x**3*S + A*y**0.5) / mu**0.5
    return C,S,x,y,tof_new

def get_tof_error(z, A, tof, r1_norm, r2_norm, mu):
    C,S,x,y,tof_new = get_vars(z,A, r1_norm, r2_norm, mu)
    return abs(tof-tof_new)/tof

def dtdz(z,A,tof,r1_norm,r2_norm, mu):
    # print("z: {}\nA: {}\ntof: {}\nr1_norm: {}\nr2_norm: {}\n:mu {}".format(z,A,tof,r1_norm,r2_norm, mu))
    C,S,x,y,tof_new = get_vars(z, A, r1_norm, r2_norm, mu)
    Sprime          = get_Sprime(z)
    Cprime          = get_Cprime(z)
    _dtdz           = (x**3.0 * (Sprime - (3.0*S*Cprime)/(2.0*C)) + (A/8.0)*((3.0*S*y**0.5)/C + A/x)) / mu**0.5
    return _dtdz

@jit
def _gauss_universal_variable_newton(mu, r1, r2, tof, rtol=1e-5, maxiter=100):
    r1_norm, r2_norm, A = prepare_solver(r1,r2)
    z = 5.0*np.pi/6.0
    ## Weird, doesn't work with the dtdx formulation, even though I've verified it???
    z = newton(get_tof_error, z, args=(A, tof, r1_norm, r2_norm, mu), fprime=None, tol=rtol, maxiter=maxiter, fprime2=None)

    C,S,x,y,tof_new = get_vars(z, A, r1_norm, r2_norm, mu)

    f         = 1 - y/r1_norm
    g         = A * (y/mu)**0.5
    gdot      = 1 - y/r2_norm
    v1 = (r2 - f*r1)/g
    v2 = (gdot*r2 - r1)/g
    return v1,v2

@jit(nopython=True)
def prepare_solver(r1,r2):
    r1_norm = np.dot(r1,r1)**0.5
    r2_norm = np.dot(r2,r2)**0.5
    ## Compute true anomaly. r1 dot r2 = r1*r2cos(nu) 
    del_nu  = 2.0*np.pi*0 + np.arccos( np.dot(r1,r2)/(r1_norm*r2_norm))
    
    ###### Evaluate Universal variable (A) and auxilary variable (y)
    A = (r1_norm*r2_norm)**0.5 * np.sin(del_nu) / (1.0 - np.cos(del_nu))**0.5
    return r1_norm, r2_norm, A


## Solve by Gauss method via universal variables
# Bate, Mueller, White: 233
## nopython makes everything MUCH faster, but also more fragile... can't use most high level functions or anything
@jit(nopython=True)
def _gauss_universal_variable(mu, r1, r2, tof, short_path=True, full_orbits=0, rtol=1e-5, maxiter=100):
    """Universal variable method
    Only formulated for a single loop right now

    Following procedure given by: http://ccar.colorado.edu/imd/2015/, and outlined in Bate, Mueller, and White. 
    """       

    ###### Prepare inputs
    if short_path:
        a = 1
    else:
        a = -1

    r1_norm = np.dot(r1,r1)**0.5
    r2_norm = np.dot(r2,r2)**0.5
    ## Compute true anomaly. r1 dot r2 = r1*r2cos(nu) 
    del_nu  = 2.0*np.pi*full_orbits + np.arccos( np.dot(r1,r2)/(r1_norm*r2_norm))
    
    ###### Evaluate Universal variable (A) and auxilary variable (y)
    A = a * ((r1_norm*r2_norm)**0.5 * np.sin(del_nu) / (1.0 - np.cos(del_nu))**0.5)

    if A == 0:
        print("Infinite number of trajectories when A==0, phase angle == 180 degrees. Cannot Compute. ")
        return None

    ###### Select trial z value
    z      = 5.0 * np.pi / 6.0
    z_low  = -4*np.pi**2
    z_high = 4*np.pi**2

    ## Homegrown bisection method
    ##      Ended up looking a lot like from: https://github.com/poliastro/poliastro/blob/master/src/poliastro/iod/vallado.py
    iternum = 0.0
    while iternum < maxiter:
        y_iter = 0
        C = get_C(z)
        S = get_S(z)
        y = r1_norm + r2_norm - A*((1-z*S)/C**0.5)
        if A > 0 and y < 0:
            # Readjust z until y > 0.0
            # Translated directly from Vallado
            while y < 0.0:
                z_low = z
                z = (0.8 * (1.0 / get_S(z)) * (1.0 - (r1_norm + r2_norm) * get_C(z)**0.5) / A)
                y = r1_norm + r2_norm - A*((1-z*S)/C**0.5)
                if y_iter > 1000:
                    # print("Y is less than zero and z value could not be adjusted")
                    z = 0
                    y = r1_norm + r2_norm - A*((1-z*S)/C**0.5)
                    break
                y_iter+=1

        x = (y/C)**0.5
        tof_new = (x**3*S + A*y**0.5) / mu**0.5

        if (abs(tof-tof_new)/tof) < rtol:
            break
        iternum += 1
        ## bisection
        if tof >= tof_new:
            z_low = z
        else:
            z_high = z
        z = (z_low + z_high)/2
    else:
        # print("Maximum number of iterations reached")
        return None 
            # raise RuntimeError("Maximum iterations {} reached, time-of-flight convergence is at {}. z: {}".format(maxiter, abs(tof-tof_new), z))

    # ###### Done! Compute actual values for v1,v2
    # summary = "Solution converged in %f iterations! z value: %f, True tof: %f, converged: %f" % (iternum, z, tof,tof_new)
    f         = 1 - y/r1_norm
    g         = A * (y/mu)**0.5
    gdot      = 1 - y/r2_norm
    v1 = (r2 - f*r1)/g
    v2 = (gdot*r2 - r1)/g
    return v1,v2

## Sometimes I hate these little helper functions...
def _get_planet(planet):
    if isinstance(planet,(str)):
        try:
            planet = getattr(ephem, planet.upper())
        except:
            utils.printRed("Planet body value <{}> was not a numeric Spice kernel identifier or a known string. Cannot continue".format(planet))
            return None
    elif isinstance(planet,(int,float)):
        try:
            planet_str = spiceypy.bodc2s(planet)
            planet = getattr(ephem, planet.upper())
        except:
            utils.printRed("Planet body value <{}> was not a numeric Spice kernel identifier or a known string. Cannot continue".format(planet))
            return None
    else:
        utils.printRed("Planet body value <{}> was not a numeric Spice kernel identifier or a known string. Cannot continue".format(planet))
        return None
    return planet

def _to_Time(epoch):
    """Do everything we can to turn what you input into a Time object"""
    if isinstance(epoch, (int,float)):
        try:
            epoch = time.Time(epoch, format="jd")
        except:
            utils.printRed("Epoch Time numeric <{}> could not be converted into an actual time quantity. Examine format ane retry".format(epoch))
            return None
    else:
        try:
            epoch = time.Time(epoch)
        except:
            try:
                epoch = pd.to_datetime(epoch)
                epoch = time.Time(epoch)
            except:
                utils.printRed("Epoch Time string <{}> could not be converted into an actual time quantity. Examine format ane retry".format(epoch))
                return None
    return epoch


def planet_ephem(planet, epoch):
    planet = _get_planet(planet)
    if planet is None:
        raise Exception("Planet specifier {} not known.".format(epoch))
    epoch = _to_Time(epoch)
    if epoch is None:
        raise Exception("Epoch given <{}> was not auto convertable into a datetime or Time object".format(epoch))
    r,v = ephem.planet_ephem(planet,epoch)
    return r,v, planet, epoch

##############################################################################
#                               Main Functions
#----------*----------*----------*----------*----------*----------*----------*
def find_trajectory(mu, r1, r2, tof, short_path=None, full_orbits=0, rtol=1e-5, maxiter=100, method="gauss-uv"):
    """Given two positions in space, the central attractor gravatational chararistics, and a time of flight, 
    figure out departure and arrival velocities
    """

    ## Make sure all units are properly converted
    if isinstance(mu,Quantity):
        mu  = mu.to(u.km**3 / u.s**2).value
    if isinstance(r1,Quantity):
        r1  = r1.to(u.km).value
    if isinstance(r2,Quantity):
        r2  = r2.to(u.km).value
    if isinstance(tof,(Quantity,time.Time)):
        try:
            tof = tof.to(u.s).value 
        except:
            raise Exception("Time of flight value provided {} could not be converted to seconds. Type: {}".format(tof,type(tof)))

    if tof <= 10000:
        # utils.printRed("TOF {} too short Not calculating trajectory".format(tof))
        return None,None #[3,3,3]*(u.km/u.s) , [3,3,3]*(u.km/u.s)
    if method == "gauss-uv":
        if short_path is None:
            # guessing... make smarter function maybe? 
            if np.cross(r1,r2)[2] > 0:
                short_path = True
            else:
                short_path = False
            print("Guessing short vs long path should be: {}".format(short_path))

        v_pair = _gauss_universal_variable(mu, r1, r2, tof, short_path=short_path, full_orbits=full_orbits, rtol=rtol, maxiter=maxiter)

    if v_pair is None:
        return None, None

    return v_pair[0] * (u.km/u.s), v_pair[1] * (u.km/u.s)

def interplanetary_trajectory(departure_body, arrival_body, departure_time, arrival_time, norms=False, short_path=None):
    """Use the Sun as the central attractor, get trajectory between two planetary bodies.

    Must be defined within the spice kernel 'de421'
    """
    r_depart, v_depart_planet, _, departure_epoch = planet_ephem(departure_body, departure_time)
    r_arrive, v_arrive_planet, _, arrival_epoch = planet_ephem(arrival_body, arrival_time)

    # If sign(cross(v_depart,v_arrive)) > 0 and (delta something anomaly)>pi, shortpath = False
    tof = arrival_epoch - departure_epoch

    v_depart,v_arrive = find_trajectory(Sun.k, r_depart, r_arrive, tof, short_path=short_path)
    if v_depart is None:
        return None, None
    
    dv_depart = v_depart.to(u.km/u.s) - v_depart_planet.to(u.km/u.s)
    dv_arrive = v_arrive.to(u.km/u.s) - v_arrive_planet.to(u.km/u.s)
    if norms:
        return norm(dv_depart), norm(dv_arrive)

    return (r_depart, v_depart_planet.to(u.km/u.s), v_depart), (r_arrive, v_arrive_planet.to(u.km/u.s), v_arrive), (dv_depart, dv_arrive)

interplanetary_trajectory_v = np.vectorize(interplanetary_trajectory, otypes=[np.ndarray,np.ndarray], excluded=["departure_body","arrival_body"])

def plot_basic_transfer(t_depart,t_arrive,depart_from="Earth",arrive_at="Mars",short_path=None):
    t_depart = _to_Time(t_depart)
    t_arrive = _to_Time(t_arrive)
    (r1, _, v1), (r2, _, v2), (dvd, dva) = interplanetary_trajectory(depart_from, arrive_at, t_depart, t_arrive,short_path=short_path)
    print("Trajectory costs: {} deltaV on departure, and {} deltaV on arrival".format(norm(dvd),norm(dva)))
    fig = plt.figure(figsize=(10, 10))
    ax  = fig.add_subplot(111, projection='3d')

    def plot_body(ax, r, color, size, border=False, **kwargs):
        """Plots body in axes object.

        """
        return ax.plot(*r[:, None], marker='o', color=color, ms=size, mew=int(border), **kwargs)

    ###### Build vectors of actual trajectories
    ## Extract whole orbits for bodies transfer for plotting
    xfer_times = np.linspace(t_depart.jd, t_arrive.jd, num=50)
    times_vec  = time.Time(xfer_times, format="jd")

    r_depart_body ,_, _, _  = planet_ephem(depart_from, times_vec)
    r_arrival_body ,_, _, _ = planet_ephem(arrive_at, times_vec)
    r_trans                 = np.zeros_like(r_depart_body)
    r_trans[:,0]            = r1
    xfer_orbit_depart       = Orbit.from_vectors(Sun, r1, v1, t_depart)
    xfer_orbit_arrive       = Orbit.from_vectors(Sun, r2, v2, t_arrive)

    for ii in range(1, len(xfer_times)):
        tof = (xfer_times[ii] - xfer_times[0]) * u.day
        r_trans[:,ii] = xfer_orbit_depart.propagate(tof,rtol=1e-6).r

    jd_init     = (t_arrive - 1.5*u.year).jd
    jd_vec_rest = np.linspace(jd_init, t_depart.jd, num=50)

    times_rest      = time.Time(jd_vec_rest, format='jd')
    r_depart_body_rest, _ = ephem.planet_ephem(_get_planet(depart_from), times_rest)
    r_arrival_body_rest, _  = ephem.planet_ephem(_get_planet(arrive_at), times_rest)

    # Color!
    color_depart0 = '#3d4cd5'
    color_departf = '#525fd5'
    color_arrive0 = '#ec3941'
    color_arrivef = '#ec1f28'
    color_sun     = '#ffcc00'
    color_orbit   = '#888888'
    color_trans   = '#444444'

    # Plotting orbits is easy!
    ax.plot(*r_depart_body.to(u.km).value, c=color_depart0)
    ax.plot(*r_arrival_body.to(u.km).value, c=color_arrive0)
    ax.plot(*r_trans.to(u.km).value, c=color_trans)

    ax.plot(*r_depart_body_rest.to(u.km).value, ls='--', c=color_orbit)
    ax.plot(*r_arrival_body_rest.to(u.km).value, ls='--', c=color_orbit)

    # But plotting planets feels even magical!
    plot_body(ax, np.zeros(3), color_sun, 16)

    plot_body(ax, r1.to(u.km).value, color_depart0, 8)
    _r1 = r_depart_body[:,0].to(u.km).value
    _rnext = r_depart_body[:,1].to(u.km).value
    _rx = r_depart_body[:,5].to(u.km).value
    _v1 = v1.to(u.km/u.s).value
    a   = Arrow3D([_r1[0], _rx[0]], [_r1[1], _rx[1]], [_r1[2], _rx[2]], mutation_scale=10, arrowstyle='-|>', color=color_depart0)
    ax.add_artist(a)
    plot_body(ax, r_depart_body[:, -1].to(u.km).value, color_departf, 8)

    plot_body(ax, r_arrival_body[:, 0].to(u.km).value, color_arrive0, 8)
    plot_body(ax, r2.to(u.km).value, color_arrivef, 8)

    # Add some text
    ax.text(-0.75e8, -3.5e8, -1.5e8, "Interplanetary trajectory:\nfrom {} to {}".format(depart_from,arrive_at), size=20, ha='center', va='center', bbox={"pad": 30, "lw": 0, "fc": "w"})
    ax.text(_r1[0] * 1.4, _r1[1] * 0.4, _r1[2] * 1.25, "{} at launch\n({})".format(depart_from, t_depart.datetime.strftime("%b %Y")), ha="left", va="bottom", backgroundcolor='#ffffff',size=7)
    ax.text(_rnext[0] * 0.7, _rnext[1] * 1.1, _rnext[2], "{} at arrival\n({})".format(arrive_at,t_arrive.datetime.strftime("%b %Y")), ha="left", va="top", backgroundcolor='#ffffff',size=7)
    ax.text(-1.9e8, 8e7, 0, "WRONGPLACETransfer\norbit", ha="right", va="center", backgroundcolor='#ffffff',size=7)
    # # Tune axes
    # ax.set_xlim(-3e8, 3e8)
    # ax.set_ylim(-3e8, 3e8)
    # ax.set_zlim(-3e8, 3e8)

    # And finally!
    ax.view_init(30, 260)

    plt.show(block=False)
    return r_trans
# http://stackoverflow.com/questions/29188612/arrows-in-matplotlib-using-mplot3d
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def Rx(phi):
    return np.array([[1, 0, 0],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi), np.cos(phi)]])

def Ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def Rz(psi):
    return np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])

def porkchop(departure_body, arrival_body, departure_start, departure_end, arrival_start, arrival_end, num=50, fillContour=True, short_path=False):
    """Always want to make sure it is correct:

    Looks similar: http://degenerateconic.com/wp-content/uploads/2014/11/pork_chop2.png
    Looks sorta similar, but one lobe is flipped? http://physics.stackexchange.com/questions/123029/why-is-there-a-gap-in-porkchop-plots
    """
    departures = [time.Time(d,format="jd") for d in np.linspace(time.Time(departure_start).jd,time.Time(departure_end).jd, num)]
    arrivals   = [time.Time(d,format="jd") for d in np.linspace(time.Time(arrival_start).jd,time.Time(arrival_end).jd, num)]
    vd = np.zeros((len(departures),len(arrivals)))
    va = np.zeros((len(departures),len(arrivals)))
    va2 = np.zeros((len(departures),len(arrivals)))
    idx = 0
    for d in departures:
        sys.stdout.write("\rExamining trajectories ==> {progress}%".format(progress=int(idx/len(departures)*100.0)))
        sys.stdout.flush()
        dv,av = interplanetary_trajectory_v(departure_body, arrival_body, d, arrivals, norms=True, short_path=True)
        dv2,av2 = interplanetary_trajectory_v(departure_body, arrival_body, d, arrivals, norms=True, short_path=False)
        vd[idx] = dv
        va[idx] = av
        va2[idx] = av2
        idx += 1
    fig,ax = plt.subplots()
    levels = np.linspace(0,25,30)
    if fillContour:
        c = plt.contourf([D.to_datetime() for D in departures] ,[A.to_datetime() for A in arrivals], va, levels)
        c2 = plt.contourf([D.to_datetime() for D in departures] ,[A.to_datetime() for A in arrivals], va2, levels)
    else:
        c = plt.contour([D.to_datetime() for D in departures] ,[A.to_datetime() for A in arrivals], np.transpose(va)**2, levels, linewidths=2)
        c2 = plt.contour([D.to_datetime() for D in departures] ,[A.to_datetime() for A in arrivals], np.transpose(va2)**2, levels, linewidths=2)


    plt.colorbar(c)
    plt.clabel(c, inline=1, fmt='%1.1f', colors='k',fontsize=11)
    plt.clabel(c2, inline=1, fmt='%1.1f', colors='k',fontsize=11)
    plt.grid()
    fig.autofmt_xdate()
    plt.title("Arrival DV for Interplanetary xfer {} ==> {}".format(departure_body,arrival_body))
    plt.show(block=False)
    return departures, arrivals, vd,va



#--------------------MOVE TO UTILITY AREA???--------------------
def mean_motion(mu, semimajor_axis):
    """Matches Tapley's notebook

    Returns angular rate Quantity in radians per second
    """

    # Put these checks into functions...
    if isinstance(mu,Quantity):
        mu  = mu.to(u.km**3 / u.s**2).value
    if isinstance(semimajor_axis,Quantity):
        semimajor_axis = semimajor_axis.to(u.km).value

    mm = ((mu/semimajor_axis)**0.5 / semimajor_axis)
    return mm * (u.rad/u.s)

#----Oblateness perturbations
def asc_node_precession_rate(j2, r_central, mu_central, semimajor_axis, inclination, eccentricity):
    """ Angular rate of change of the precession of the ascending node
        Reference: https://en.wikipedia.org/wiki/Nodal_precession
    """
    # Do we actually want to convert???
    mm = mean_motion(mu_central, semimajor_axis)

    if isinstance(mu_central,Quantity):
        mu  = mu_central.to(u.km**3 / u.s**2).value
    if isinstance(semimajor_axis,Quantity):
        semimajor_axis = semimajor_axis.to(u.km).value
    if isinstance(r_central,Quantity):
        r_central = r_central.to(u.km).value
    if isinstance(inclination,Quantity):
        inclination = inclination.to(u.rad).value
    if isinstance(eccentricity,Quantity):
        eccentricity = eccentricity.to(u.rad).value
    
    # j2 is unitless

    ## First, get the mean motion in rad/s

    prec_rate = -(1.5) * mm * j2 * (r_central/semimajor_axis)**2 * np.cos(inclination) * (1-eccentricity)**-2
    return prec_rate

def get_sun_sync_inclination(j2, r_central, mu_central, semimajor_axis, eccentricity):
    # Function of just inclination
    def pres_rate_of_inclination(inclination):
        return asc_node_precession_rate(j2,r_central,mu_central, semimajor_axis, inclination, eccentricity).value - (2*np.pi / (86400*365.25))

    ss_inc = newton(pres_rate_of_inclination, 1)
    return ss_inc * u.rad

def test():
    """Not consolodted test function, just a place to copy/paste interactive test case
    """
    ## From http://nbviewer.jupyter.org/github/poliastro/poliastro/blob/master/examples/Going%20to%20Mars%20with%20Python%20using%20poliastro.ipynb
    r_earth = [64601872, 1.2142001e8, 52638008] * u.km
    r_mars  = [-1.2314831e8, 1.9075313e8, 90809903] * u.km

    ## Not a valid funciton, just writing down some useful starting point tests
    import astropy.units as u
    from astropy import time

    from poliastro import iod # This is the big one!
    from poliastro.bodies import Sun
    from poliastro.twobody import Orbit
    from poliastro import ephem
    ephem.download_kernel("de421")
    ## MSL Stats from: http://mars.jpl.nasa.gov/msl/mission/overview/
    launch_date  = time.Time("2011-11-26 15:02:00",scale="utc")
    arrival_date = time.Time("2012-08-06 05:17:00",scale="utc")
    tof = arrival_date - launch_date
    print("Time of flight: {} hours".format(tof.to(u.h)))
    ## Get vector of times from launch and arrival Julian days
    N          = 50
    launch_jd  = launch_date.jd
    arrival_jd = arrival_date.jd
    jd_vec     = np.linspace(launch_jd, arrival_jd, num=N)

    times_vec  = time.Time(jd_vec, format="jd")
    ## Use `ephem` module to get Earth and Mars positions
    r_earth, v_earth = ephem.planet_ephem(ephem.EARTH, times_vec)
    r_mars, v_mars   = ephem.planet_ephem(ephem.MARS, times_vec)
    r0 = r_earth[:,0]
    r1 = r_mars[:,-1]

    ## Compute departure/arrival velocities, using Sun as main attractor
    (v_depart_iod, v_arrive_iod), = iod.lambert(Sun.k, r0, r1, tof)
    vd,va = lambert._gauss_universal_variable(Sun.k.value, r0.value, r1.value, tof.to(u.s).value,rtol=1e-8,finder=1,maxiter=1000000)


    departures=[time.Time(d,format="jd") for d in np.linspace(time.Time("2020-01-01").jd,time.Time("2020-04-01").jd)]
    arrivals=[time.Time(d,format="jd") for d in np.linspace(time.Time("2020-08-01").jd,time.Time("2021-04-01").jd)]
    xx=np.zeros((50,5))

    idx=0
    for d in departures:
        x,y=lambert.interplanetary_trajectory("Earth","mars",d,arrivals)
        xx[idx] = x
        idx+=1
    c = contour([D.value for D in departures] ,[A.value for A in arrivals],xx)
    clabel(c, inline=1, fontsize=10)
















