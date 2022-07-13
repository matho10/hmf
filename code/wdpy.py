# -*- coding: utf-8 -*-
"""
useful python functions.
This library contains useful computational functions!
Table

Created on Sun Sep 22 11:04:01 2013

@author: Matho Di Wang,
"""
import scipy as sp
import numpy as np

import scipy.optimize as opt
from random import uniform
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import itertools as itr
from matplotlib import pyplot as plt

def take_list(x, idx):
    """ Returns sublist of x according to a list of index.
    Parameters
    ----------
    x : list
        what list you want to take from?
    idx : list
        the list of index.

    Examples
    --------
    $ take_list([1,2,3],[0])
     [1]
    $ take_list([1,2,3],[0,0,1])
     [1,1,2]
    """
    return [x[i] for i in idx]



class interp:
    """ Provide a convenient treatment to multidimentional (including 1-d) interpolation
    problem. By first defining the class object, z = interp(points, values). 
    z(x) returns the interpolated values at x. Here, x may be a list containing 
    all interested points. 
        method = 'linear' or 'cubic'.
        
    """
    def __init__(self, points, values):
        self.points, self.values = points, values
        if isinstance(self.points[0], (int, float)):
            self.dim= 1
        else:
            self.dim = len(self.points[0])
    def __call__(self, z, method='cubic'):
        if self.dim >= 2 and isinstance(z[0], (int, float)):
            z = tuple(z)
        value = griddata(self.points, self.values, z, method)
        if np.shape(value)==():
            return float(value)
        return value
    

class LinInterp:
    """ Provide linear interpolation in one dimension
    example:
    xx = np.linspace(0, 2, 200)
    yy =  xx**2 + sp.randn(size(xx))
    f = LinInterp(xx, yy)
    print(f(pi/2))
    
    """
    
    def __init__(self,X,Y):
        self.X, self.Y = X,Y
    def __call__(self,z):
        if isinstance(z,int) or isinstance(z, float):
            return sp.interp([z], self.X, self.Y)[0]
        return sp.interp(z, self.X, self.Y)

def scaleback(x,z=None,axis=None):
    """ scaleback(matrix, z= 1, axis=0) returns a new matrix b, such that
        b.sum(axis) == z
    """
    if z==None:
        z=1
    if axis==None:
        axis=0
    
    sha=list(x.shape)
    if len(sha)==1:
        return x/sum(x)*z
    else:
        sha[1]=1
    r=x.sum(axis)/np.array(z)
    
    if axis==0:
        M=np.tile(r,sha)
    else:
        r=r.reshape(sha[0],sha[1])
        M=np.tile(r,[sha[1],sha[0]])

    output=x/M
    return output

class ecdf:
    """
    Empirical cumulative distribution function of 1-d random variable.\n
    Example:\n
    samples=np.random.uniform(0,1,2000) \n
    F=ecdf(samples) \n
    F(0.5) \n
    F.plot() \n

    """
    def __init__(self,obs):
        obs=np.array(obs)
        obs.sort()
        self.obs=obs
        
    def __call__(self,x):
        return np.mean(self.obs<=x)
        
    def plot(self,ra=None):
        if ra==None:
            ra=[self.obs.min(), self.obs.max()]
        a,b=ra[0], ra[1]
        carry=np.array(range(len(self.obs)))+1
        c=carry/float(carry.max())
        conditions=(self.obs>=a) * (self.obs<=b)
        y=c[conditions]
        x=self.obs[conditions]
        plt.plot(x,y)
        
def fsolve(f,x0=0):
    """ Generic function=0 solver. This is only for one-argument function. 
    fsolve(f,x0=0)
    f: a function
    x0: [a,b] or x0: fsolve will take care of it automatically.
    """
    if isinstance(x0,(int,float)):
        return opt.newton(f,x0)
    else:
        return opt.brentq(f,x0)
        
def sample1(phi):
    """ returns i with probability phi[i], where phi is an array. """
    a=0
    U=uniform(0,1)
    for i in range(len(phi)):
        if a <U <= a+phi[i]:
            return i
        a=a+phi[i]

def sample(phi=None, x=None, m=1, rep=True):
    """ sample(phi, x=range(len(phi)), m=1, rep=True) returns stochastic 
    sample of x with or without replacement
    x: population: default is range(len(phi))
    phi: weight: default is a 1-vector indicating equal weights
    m: sample size
    rep: with replacement, defaut is True
    
    """
    if phi==None and x==None:
        print("too few information. returns 0")
        return 0
    if phi==None:
        phi=np.ones(len(x))
    if x==None:
        x=range(len(phi))
    phi = scaleback(np.array(phi), 1)
    res=[]
    if rep:
        if m==1:
            return x[sample1(phi)]
        for j in range(m):
            y=sample1(phi)
            res.append(x[y])
        return res
    else:
        if m>len(phi):
            print("error! Without rep, one can only generate a shorter list.")
            m = len(phi)
        for j in range(m):
            y=sample1(phi)
            res.append(x[y])
            phi[y]=0
            phi=scaleback(np.array(phi), 1)
        return res
    
def cbind(x,y):
    shp=x.shape
    return np.hstack((x,y.reshape(shp[0],1)))
    
def rbind(x,y):
    shp=x.shape
    return np.vstack((x,y.reshape(1,shp[1])))
    
def sample_rank(x):
    """ y=sample_rank(x)  returns an array of ranks, with the same size as vector
    x
    """
    integer=np.vectorize(int)
    y= [sum(integer(x>=x0)) for x0 in x]
    return np.array(y)

def inverse_func(f):
    """ x=inverse_func(f) returns the inverse function: f^-1, such that x(y) = 
    solution to equation: f(x) = y
    """
    def g(y):
        # g(y) solves equation: f(x)-y==0.
        eq = lambda x: f(x)-y
        return fsolve(eq, x0=0)
    return g

def ezplot(f, interval=[-1,1], N=1000):
    """quickly plot a function handle f, given interval specified. 
    """
    
    xx= np.linspace(interval[0], interval[1], N)
    yy = map(f, xx)
    plt.plot(xx, yy)

def D(f, right=True, dx= 1e-6):
    """ D(f, right=T) returns the 1-D right approximate derivative of function f,
    as a function
    
    example: 
    f = lambda x: sin(x)
    g = user.D(f)
    user.ezplot(g, [-1,1])
    """
    if right:
        g=lambda x: (f(x+dx)-f(x))/dx
    else:
        g=lambda x: (f(x)-f(x-dx))/dx
    return g

def index_min(values):
    return min(range(len(values)),key=values.__getitem__)

def index_max(values):
    return max(range(len(values)),key=values.__getitem__)

class rv1(object):
    """ rv1() defines a 1-d random variable in its theoretical form.
    
    Objects:
    mode: either "continuous" or "discrete". 
    F: cdf of random variable.
    p: pdf (if "continuous") or pmf (if "discrete")
    X: a list of all possible states. If the mode is continuous, it should be a 
        list [lower bound, upper bound]
        if the mode is discrete, it should be a vector containing all possible 
        values.
    
    """
    
    
    def __init__(self, F=None, mode = 'continuous', x=None):
        pass

def sorted_according_to(x: list, according_to: list):
    """

    :param x: what list you want to sort?
    :param according_to: according to what you want to sort x?
    :return: a sorted list of x
    """
    return [x for _,x in sorted(zip(according_to,x))]


class num_fun:
    """
    numerical approximation of a function, used to do post-work after you map function on some grids.
    """
    def __init__(self, x:list, y:list):
        """

        :param x: input locations.. a list.
        :param y: function values on these corresponding points.
        :returns:

        """
        self.x = x
        self.y = y
    def min_sorted(self, top=1, output='(location, func_value)'):
        """
        Very inefficient function minimizer, which supports to distill several minimization point, but very
        flexible.

        :param top: How many points you want to select? by default only produce the minimum (top = 1)
        :param output: '(index, func_value)', '(position, func_value)', 'position', 'index', 'func_value', all support!
        :return:
        """
        sorted_pairs = sorted(zip(self.y, self.x, list(range(len(self.x)))))[0:top]
        index = [var[2] for var in sorted_pairs]
        location = [var[1] for var in sorted_pairs]
        func_value = [var[0] for var in sorted_pairs]
        return eval(output)


def mix_map(list_from, list_to, test_func = None, value_func = None):
    """

    Parameters
    ----------
    list_from : list
        a list from which you want to select, or run as base list
    list_to : list
        a list (target)
    test_func : function
        None by default. If this one is not none, then return a sublist of list_from
        such that test_func(list_from, all list_to) == True.
    value_func : function
        None by default. If this one is not none and test_func is none, then return
        a list like ((list_from, list_to, value_func(list_from, list_to))


    Returns
    -------

    """
    if test_func is not None:
        true_table = [all([test_func(a, b) for b in list_to]) for a in list_from]
        return_list = [a for (a,_) in list(zip(list_from, true_table)) if _]
        return return_list
    if value_func is not None:
        mapped_list = [(a, b, value_func(a, b)) for a in list_from for b in list_to]
        a_list = [a for a,_,_ in mapped_list]
        b_list = [b for _,b,_ in mapped_list]
        c_list = [c for _,_,c in mapped_list]
        return (a_list, b_list, c_list)

