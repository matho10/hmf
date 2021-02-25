# -*- coding: utf-8 -*-
# I use label ### to indication changes that I may delete later...

"""
Heterogeneous modelling functions, version 3.
Author: Wang Di,
Date: 2019, Sept.



This is an improved version of
hmf2 packages
including

class space(x1 = [0,1,...], x2 = [1.1, 1.2...]: **kwargs)
  .name
  .set
  .meshgrid_dict
  .N
  .number
  .number_matrix
  -index_to_number
  -number_to_index

class anonymous_space(x1 = [], x2=[], ...: *args)


class optimization(problem ='max', state = dict, action = dict, constrain =
string, u = string)

after defining, use,
optimization.feasiblity()
optimization.optimize()

"""

# === Setup environment
import numpy as np
import scipy as sp
from numba import vectorize
from numba import jit
import quantecon as qe
from scipy import sparse
import numdifftools as nd
import pandas as pd
from pathos.pools import ProcessPool
import wdpy
import itertools as itr

def closed_range(start, stop, step=1):
    dir = 1 if (step > 0) else -1
    return range(start, stop + dir, step)


# ------- Some foo functions, use them to occupy optim class when you do simple tasks ---------
def foo():
    return None


def foo_true(**kwargs):
    return True


def foo_1(**kwargs):
    return 1



class space:
    ''' sno = space(x1 = ..., x2=..., ...)
    sno.number(a vector of indexies)
    sno.index(a number)

    仍有一些问题：1. 是否要让space默认计算self.value ?
    '''

    def __init__(self, **kwargs):
        self.dict = kwargs
        self.name = list(kwargs.keys())
        self.set = list(kwargs.values())
        # self.grid = np.meshgrid(*self.set, indexing = 'ij') # returns meshgrids
        grid = np.meshgrid(*self.set, indexing='ij')
        self.meshgrid_dict = dict(zip(self.name, grid))  # meshgrid_dict is a dictionary contains all grids, with names!
        self.shape = [len(values) for values in
                      kwargs.values()]  # shape of state space: notice that this is not the shape of gridded matrix!!!! I made a mistake when firstly code.
        if len(kwargs)==0:
            self.N = 0
            self.number = []
            self.number_matrix = []
        else:
            self.N = grid[0].size  # the length of all state possibles, length of space numbers
            self.number = np.array(range(self.N))
            self.number_matrix = self.number.reshape(self.shape)
        # generate .index, .value,
        self.index = self('index')
        self.value = self('value')


    def index_to_number(self, index):
        return (self.number_matrix[tuple(index)])

    def number_to_index(self, number):
        ff = np.array(np.where(self.number_matrix == number)).T[0]
        return (list(ff))

    def index_to_value(self, index):
        '''index_to_value(self, index like [2, 5, 3]) -> [3.1, 5.2, 1.1] the value
         of variables, given the index in each variable's set.
        '''
        # state_with_index is like [[0,1,2,...,number of variables], [3, 1, ...]].T
        # Thus, each row is a (i, index_of_i-th variable = j)
        state_with_index = np.array([range(len(self.set)), index]).T
        return [self.set[i][j] for (i, j) in state_with_index]

    def number_to_value(self, number):
        return self.index_to_value(self.number_to_index(number))

    def __call__(self, method="number"):
        if method == "number":
            return self.number
        elif method == "index":
            return [self.number_to_index(i) for i in self.number]
        elif method == "value":
            return [self.index_to_value(self.number_to_index(i)) for i in self.number]
    def map(self, func, full_output=False):
        # Legacy
        input_list = self.value
        res = list(map(func, input_list))
        self.v = res
        if full_output:
            return (input_list, res)
        else:
            return res
    def named_map(self, func, **par):
        '''
        :param func: named function, its argument order must be the same with how space is defined!
        named_map allows function to take in parameters
        e.g. f(x, y, theta) where x, y are in space s, and theta is parameter.
        To get all result of f(x, y), use s.named_map(f, theta = ..)
        :return: [list of function values]
        '''
        su = lambda args: func(*args)
        input_state_list = self.value
        input_list = list(tuple(input_state_list[i]) + tuple(par.values()) for i in self.number)
        res = list(map(su, input_list))
        self.value = input_state_list
        self.v = res
        return res

    def imap(self, func, nodes = 8, full_output=False):
        '''
        :param func: single-input function([value]) to map on all values, e.g. f should be something such that f([0,1,2]) is feasible. If you define
        the function input as a scaler, e.g. f(1) works, but not f([1]), then you need to do g=lambda x: f(x[0]) then
        use self.imap(g). Similarly, if your function defined as f(a=1, b=2) you need to transform it into:
        g= lambda x: f(x[0], x[1]). Use the correctly ordered space: s= space(a = np.linspace(...), b=...) This works as
        s.imap(g)
        :param nodes=8: how mnay nodes used in computation
        :full_output = False: if True, then returns ([all locations]=self('value'), [func(locations)])
        if False, only return ([func(locations)])
        
        '''
        input_list = self.value
        pool = ProcessPool(nodes = nodes)
        res = list(pool.imap(func, input_list))
        self.v = res
        if full_output:
            return (input_list, res)
        else:
            return res
    def slice(self, input_type = 'index', **kwargs):
        '''
        NOT FINISHED.
        想法：1 作出 [a,b,c] 所有的， 这样的list （for b=b0, c=c0, a=any)
        2. 对于所有self('index') 进行遍历，每个看看是否由1 中的list，若有的话，报告此时的self.v[i]
        3. 整理汇报结果。
        After having computed self.v, we want to know how v varies with one single variable,
        given other variables are fixed.
        :param kwargs:  e.g., a=1, b=2, ...
        :param input_type: 'index'/'value' currently only support 'index'
        :return: [x, y (function values) ]

        '''
        if kwargs == None:
            return (self('value'), self.v)



class anonymous_space:
    """
    Anonymous Space Object creates joint space of anonymous variables, 
    __init__(*args):
        args are unnamed np.array objects created by, for example, np.linspace
    """
    def __init__(self, *args):
        self.set = args
        # self.grid = np.meshgrid(*self.set, indexing = 'ij') # returns meshgrids
        grid = np.meshgrid(*self.set, indexing='ij')
        self.grid = grid  # grid is a dictionary contains all grids, without names!
        self.shape = [len(values) for values in self.set]  # shape of state space: notice that this is not the shape of gridded matrix!!!! I made a mistake when firstly code.
        self.N = grid[0].size  # the length of all state possibles, length of space numbers
        self.number = np.array(range(self.N))
        self.number_matrix = self.number.reshape(self.shape)

    def index_to_number(self, index):
        return (self.number_matrix[tuple(index)])

    def number_to_index(self, number):
        ff = np.array(np.where(self.number_matrix == number)).T[0]
        return (list(ff))

    def index_to_value(self, index):
        '''index_to_value(self, index like [2, 5, 3]) -> [3.1, 5.2, 1.1] the value
         of variables, given the index in each variable's set.
        '''
        # state_with_index is like [[0,1,2,...,number of variables], [3, 1, ...]].T
        # Thus, each row is a (i, index_of_i-th variable = j)
        state_with_index = np.array([range(len(self.set)), index]).T
        return [self.set[i][j] for (i, j) in state_with_index]

    def number_to_value(self, number):
        return self.index_to_value(self.number_to_index(number))

    def __call__(self, method="number"):
        if method == "number":
            return self.number
        elif method == "index":
            return [self.number_to_index(i) for i in self.number]
        elif method == "value":
            return [self.index_to_value(self.number_to_index(i)) for i in self.number]
    def imap(self, func, nodes = 8, full_output=False):
        '''
        :param func: single-input function([value]) to map on all values, 
        :param nodes: 8, how mnay nodes used in computation
        :full_output: False, if True, then returns ([all locations]=self('value'), [func(locations)])
        if False, only return ([func(locations)])
        
        '''
        input_list = [self.index_to_value(self.number_to_index(i)) for i in self.number]
        pool = ProcessPool(nodes = nodes)
        res = list(pool.imap(func, input_list))
        if full_output:
            return (input_list, res)
        else:
            return res

class points:
    """
    points collect several individual points in the Euclidean space.
    Attribute
    ---------
    value : list
        location of these points
    arm : list
        describe the closure on each points.


    """
    def __init__(self, location: list = None, arm: list = None, func_value = None):
        """

        Parameters
        ----------
        location : list
            a list of np.array gives the locations of these points
        arm : list
            closure of each points along dimensions, e.g. [0.5, 0.3]
        func_value : list
            some function value we assign to points.
        """
        self.location = location
        self.arm = arm
        if func_value is None:
            self.func_value = [np.nan] * len(self.location)
        else:
            self.func_value = func_value
        self.dim = len(self.location[0])
    def verge(self, repeat = 2):
        """
        Parameters
        ----------
        repeat : int
            how many times you want to repeat the verge point search?
        Returns
        -------
        list
            a list of verge points.
        """
        # this is the set we want to finally return
        res_points = []

        # define plus function
        def plus(a,b):
            return np.array(a)+np.array(b)

        def far_enough(a, b):
            return np.sqrt(sum((np.array(a)-np.array(b))**2))>max(self.arm)/1000

        # creates the add_loc to add on each point
        sep_arm = []
        for i in range(len(self.arm)):
            sep_arm.append([0, self.arm[i], -self.arm[i]])
        add_loc = list(itr.product(*sep_arm))

        from_where = self.location.copy() # creates the first from_where point collection
        # from these points, we want to perform proliferation tasks.
        for j in range(repeat):
            a1, a2, a3 = wdpy.mix_map(from_where, add_loc, value_func=plus)
            from_where = wdpy.mix_map(a3.copy(), from_where, test_func=far_enough)
            from_where = wdpy.mix_map(from_where, self.location, test_func = far_enough)
            # 问题：这里可能还会有重复！需要重新检查。需要再比对from_where 与res_points 的距离么？
            # 需要！
            # 另一个问题：不同点生成的端点可能会重复！
            if j>= 1:
                from_where = wdpy.mix_map(from_where, res_points, test_func = far_enough)
            res_points += from_where
        return list(np.unique(np.array(res_points), axis = 0))

    def imap(self, func, nodes=8):
        """

        Parameters
        ----------
        func : function
            single-input function([value]) to map on all values,
        nodes : int
            number of computational nodes, 8 is default

        Yields
        ------
            func_value

        """
        input_list = self.location
        pool = ProcessPool(nodes=nodes)
        res = list(pool.imap(func, input_list))
        self.func_value = res

    def min_sorted(self, top=1, output='(location, func_value)'):
        """
        Very inefficient function minimizer, which supports to distill several minimization point, but very
        flexible.

        :param top: How many points you want to select? by default only produce the minimum (top = 1)
        :param output: '(index, func_value)', '(location, func_value)', 'location', 'index', 'func_value', all support!
        :return:
        """
        #print(self.func_value)
        #print(list(zip(self.func_value, self.location, list(range(len(self.func_value))))))
        sorted_pairs = sorted(zip(self.func_value, list(range(len(self.func_value))), self.location))[0:top]
        index = [var[1] for var in sorted_pairs]
        location = [var[2] for var in sorted_pairs]
        func_value = [var[0] for var in sorted_pairs]
        return eval(output)

    def union(self, another_points_object):
        """

        Parameters
        ----------
        another_points_object: points
            another points object to unionize

        Yields
        ------
        points object

        """
        if any(np.array(self.arm) != np.array(another_points_object.arm)):
            print('Warning! .arm not equal! Using the self.arms')
        self.location += another_points_object.location
        self.func_value += another_points_object.func_value



class vertex:
    """
    LEGACY CODES, NOT USED ANYMORE. BUT .points USES IT TO INITIALIZE
    GRIDS. DO NOT DELETE.
    vertex object is a "regularized" space, less complicated,
    suitable to run a computation mapped to all vertex points.
    vertex is not equipped with index object

    Attribute
    ----------
    ranges : list of tuple
        [(1,2), (3,4)], representing the ranges of each dimensions.
    Ns : list
        [2, 3] representing number of grids along each dimension
    space : anonymous_space
        anonymous_space object of vertex points
    dim : int
        dimensions
    value : list
        a list of all positions of points
    accompany : list
        a list of all positions in the unit-cubic space. This space consists of corresponding points in a cubic.
        used to compute a normalized "norm" when "distance" is properly defined.
    max_accompany_arm : float
        = 1/ max(Ns) this reflects how "accurate" the vertex is.
    """
    def __init__(self, ranges:list = None, Ns = None):
        """
        Parameters
        ----------
        ranges : list
            e.g [(1,2), (3,4)], representing the ranges of each dimensions.
        Ns : list or a number
            e.g. [2,3] or 2(indicates [2,2])
        """
        self.ranges=ranges
        self.dim = len(self.ranges)
        if isinstance(Ns, list):
            self.Ns=Ns
        else:
            self.Ns = [Ns] * self.dim
        # create a tuple containing all sets
        sets_tuple = [np.linspace(self.ranges[i][0], self.ranges[i][1], self.Ns[i]) for i in range(self.dim)]
        self.space = anonymous_space(*sets_tuple) # sub_class as a anonymous_space
        self.value = self.space('value')
        accompanied_space = anonymous_space(*(np.linspace(0, 1, self.Ns[i]) for i in range(self.dim)))
        self.accompany = accompanied_space('value') # creates a corresponding accompanied space...
        self.max_accompany_arm = 1/max(self.Ns)
    def proliferate(self, b, method = 'fixed'):
        """

        :param b: e.g. 2, multiplier,
        :param method: 'fixed' or 'mobile', 'fixed' if preserving original points, in this case,
            Ns[t+1]= (b-1)*(Ns[t]-1) + Ns[t]
        :return: a vertex object with more densed grids
        """
        if method == 'fixed':
            return vertex(self.ranges, list( (np.array(self.Ns)-1)*( b-1) + np.array(self.Ns)))
        else:
            return vertex(self.ranges, list(np.array(self.Ns)*b))
    def imap(self, func, index_to_map = None, nodes=8, full_output=False):
        """
        using patho.multiprocessing to map a function on selected vertex.
        ...

        Parameters
        ----------
        func : function
            single-input function([value]) to map on all values
        index_to_map:
            a list representing what indice in vertex values you want the function to map on.
        nodes : int
            =8, how mnay nodes used in computation
        full_output : Boolean
            if True, then returns ([all locations]=self('value'), [func(locations)])
            if False, only return ([func(locations)])
            note: I still don't know whether it is beneficial to include this piece in our project.
            Maybe, I can do direct computations.
        Returns
        -------
        tuple
            if full_output = True, returns ([all locations]=self('value'), [func(locations)])
            if full_output = False, returns ([func(locations)])
        """
        if index_to_map is None:
            input_list = self.value
        else:
            input_list = [self.value[i] for i in index_to_map]
        pool = ProcessPool(nodes=nodes)
        res = list(pool.imap(func, input_list))
        self.imap_value = res # create a func_value input for further enquiry..
        if full_output:
            return (input_list, res)
        else:
            return res


class optim:
    def __init__(self, problem="max", state: dict = {},
                 action: dict = {},
                 constrain=None,
                 u=foo_1, par={}, index_transition=None,
                 state_space=None, action_space=None, joint_space=None, parallel = 1):
        """
        optim is the synthetic class that prepare one for computing optimizing problem. The class
        first works to define discrete space, then can be used to compute and optimize functions
        on discretized space, computing the constrained problem, and computing transition matrix
        in terms of a markov chain.
        xo = optim("max", state = {"x1": np.arange(1, 3), "x2": np.arange(3, 6), "x3": np.arange(6, 10) },
        action = {"y1": np.arange(1.1, 3.1), "y2":np.arange(3.1, 6.1)},
        constrain = feasible_func,
        u = u_func, par= {...})

        OVERHAUL, Branch from the : INTENDS TO ADD FOLLOWING FEATURES:
        1. allow empty space: i.e. allow state == {}, allow empty constrain functions.
        2. enable parallelling computation, but by default it should not be enabled.


        Attributes
        ----------
            state_space
            action_space
            joint_space : space

        Parameters
        ----------
        problem : str
            define the problem, whether it is "max" or "min"
        state : dict
            e.g. {'x1': np.linspace(3, 5, 10), 'x2': ...} describing state space
        action : dict
            describe actions
        constrain : function
            function(**state, **action, **par) return False or True
            False: outside the constrain set.
        u : function
            function(**state, **action, **par) returns the payoff.
            !!!!constrain and u are ORDERED input functions! must be defined in the order of
            state, action, pars!!!! And input is NAMED! That is, the following statement
            is correct way to define a function used here.
                def constrain(a = ..., b=..., x=..., i=...):
            where a, b are states, which should be named in the state dictionary,
            x is a action,
            i is parameter.
        par : dict
            {'i1': 3, 'i2': 5} containing valued parameters.
        index_transition : function
            function([index of state and action]) that returns
            something like[ [0.5, [... index in state space that has probability of 0.5 in next period]],
            [0.1, [...]], ....]
        parallel: 1, default is 1, how many cores used for doing constrain and utility computations.


        """
        self.problem = problem
        self.state = state
        self.action = action

        self.constrain = constrain
        self.u = u
        self.par = par
        self.index_transition = index_transition

        # forthcoming properties
        self.feasible_index = None  # dictionary containing feasible indices, each is a L array.
        self.feasible_state = None
        self.feasible_action = None
        self.v = []  # value functions a S array
        self.ρ = []  # policy functions a S array
        self.reward_grid = None  # value grid for all feasible state-action pairs.
        self.R = []  # L-array, reward for all feasible pairs of state-actions
        self.test = None
        self.Q = None  # transition matrix, S * L matrix
        self.parallel = parallel

        if state_space != None:
            self.state_space = state_space
            self.action_space = action_space
            self.joint_space = joint_space
            self.state = state_space.dict
            self.action = action_space.dict
        else:
            self.state_space = space(**state)
            self.action_space = space(**action)
            self.joint_space = space(**state, **action)

    def feasibility(self, method=None):
        '''Feasibility solves the feasible indices, and 1. produce feasible state-action paired numbers. 2. compute
        the reward matrix self.R
        self.feasibility() would update self.feasible_state => [0,1,5,7,....] consists of feasible states, paired with
        self.feasible_action => [5,2,....]
        these number can be turned back into indices, using
        self.state_space.index_to_number(5), for example...
        '''

        # print("start feasibility computation.")
        # === constrain_grid is the grid in joint_space containing feasible combinations
        #print("creating constrain grid...")
        if self.constrain == None: # if no default constrain function, use all indices.
            # first, use space('index') to call all combined indices in joint_space. To split them,
            # just transpose the matrix to use each row.
            s_index = np.transpose(np.array(self.joint_space('index')))
            # split the matrix and make them into a tuple ( consistent with the result of np.where)
            constrain_grid = tuple([i for i in s_index])
        else:
            constrain_grid = np.where(self.constrain(**self.joint_space.meshgrid_dict, **self.par))
        self.constrain_grid = constrain_grid  ###
        # === reward grid defined here, original code, very inefficient
        # because it computes all grids. The newer one only computes the feasible one
        # self.reward_grid = self.u(**self.joint_space.meshgrid_dict, **self.par)
        # self.R = self.reward_grid[tuple(constrain_grid)]

        # === This is the feasible index for each variables, containing each feasible state-action pairs
        self.feasible_index = dict(zip(self.joint_space.name, constrain_grid))

        # === Now turn these indices into state and action paired numbers
        #print("creating .feasible_state and .feasible_action, which are list of numbers ")
        state_indices = np.array([self.feasible_index[j] for j in self.state.keys()])
        self.feasible_state = np.apply_along_axis(self.state_space.index_to_number, 0, state_indices)
        self.feasible_state_indices = state_indices
        action_indices = np.array([self.feasible_index[j] for j in self.action.keys()])
        self.feasible_action = np.apply_along_axis(self.action_space.index_to_number, 0, action_indices)
        self.feasible_action_indices = action_indices
        self.L = self.feasible_state.size

        #print("creating reward grid and R list...")
        values = list(self.joint_space.set[i][self.constrain_grid[i]] for i in range(len(self.joint_space.set)))
        self.feasible_values = values
        self.feasible_input_list = list(tuple(values[i][j] for i in range(len(values))) +
                                        tuple(self.par.values()) for j in range(self.L))
        su = lambda args: self.u(*args)
        if self.parallel >=2:  # multiprocessing is still experimental, do not use it yet!
            # WRITE HERE NOW! imap su onto feasible_input_list
            pool = ProcessPool(nodes = self.parallel)
            self.R = np.array(list(pool.map(su, self.feasible_input_list)))
            pool.close()
            pool.join()
        else:
            # self.R = np.array([su(self.feasible_input_list[i]) for
            # i in range(self.L)])
            self.R = np.array(list(map(su, self.feasible_input_list)))

    def populate_Q(self):
        '''
        populate_Q() works to generate self.Q, which is a L x Nx sparse matrix,
        whose (i row, : ) is the next period distribution of state variables.
        '''
        # print("populate Q matrix")
        Q = sparse.lil_matrix((self.L, self.state_space.N))
        # a sparse transition matrix, L x Nx, assigning probability from numbered state-action
        # pairs to next period state number

        # all_index = np.concatenate((self.feasible_state_indices, self.feasible_action_indices)).T
        all_index = np.array(self.constrain_grid).T
        # a big matrix containing all index of feasible state-action pairs
        for i in range(self.L):
            # the i-th number of feasible state-action index pairs,

            # produce the next period transition list, like [[0.5, [s1, s2, s3]], ...]
            transition = self.index_transition(all_index[i, :])
            for j in range(len(transition)):
                element = transition[j]
                # specific possible scenarior, named as element, is a list like
                # [0.25, [0, 1, 5, ...]]
                # where 0.25 is probability assigned to this scenarior, and [0, 1, 5, ...]
                # is the state index of this scenarior
                number = self.state_space.index_to_number(element[1])
                # next period state
                Q[i, number] = element[0]
        self.Q = Q
        return

    def optimize(self):
        '''self.optimize(), after running (or without running self.feasibility) updates self.v and self.ρ, two vectors corresponding to
        optimized values (max or min, depending on self.problem), and
        '''
        # print("optimize...")
        sar_matrix = np.array([self.feasible_state, self.feasible_action, self.R,
                               list(range(self.L))]).T
        # this creates a state-action-reward matrix, all containing feasible combinations
        v = []
        ρ = []
        ω = []
        if self.problem == "max":
            for i in self.state_space.number:
                if i in sar_matrix[:, 0]:
                    relevant_sar = sar_matrix[sar_matrix[:, 0] == i]
                    argmax_point = relevant_sar[:, 2].argmax()
                    v.append(relevant_sar[:, 2].max())
                    ρ.append(int(relevant_sar[:, 1][argmax_point]))
                    ω.append(int(relevant_sar[:, 3][argmax_point]))
                    continue
                else:
                    v.append(np.nan)
                    ρ.append(np.nan)
                    ω.append(np.nan)
                    continue
            self.ρ = np.array(ρ)
            self.v = np.array(v)
            self.ω = np.array(ω)
        else:
            for i in self.state_space.number:
                if i in sar_matrix[:, 0]:
                    relevant_sar = sar_matrix[sar_matrix[:, 0] == i]
                    argmin_point = relevant_sar[:, 2].argmin()
                    v.append(relevant_sar[:, 2].min())
                    ρ.append(int(relevant_sar[:, 1][argmin_point]))
                    ω.append(int(relevant_sar[:, 3][argmin_point]))
                    continue
                else:
                    v.append(np.nan)
                    ρ.append(np.nan)
                    ω.append(np.nan)
                    continue
            self.ρ = np.array(ρ)
            self.v = np.array(v)
            self.ω = np.array(ω)

            # relevant_sar[:, 2]

    def optimize_transition(self):
        '''
        produces QS, Fn, ωn matrices, used in the simulation of heterogeneous agents
        Need to define Q matrix using populat_Q, with index_transition function Inputted first!

        '''
        QQ = self.Q.toarray()
        self.QS = np.array([QQ[self.ω[i], :] for i in range(self.state_space.N)])
        self.Fn = np.array(self.feasible_input_list)
        self.ωn = np.array([self.Fn[self.ω[i], :] for i in range(self.state_space.N)]).T
        # ensures that ωn[j] is the j-th variable, in the state-action-par joint space!
        # np.dot(ωn[j], X) is the population mean of ωn[j], where X is the distribution
        # vector.
        return

    def optimize_all(self):
        self.optimize()
        self.optimize_transition()
        return

    def all(self):
        '''
        Execute all methods one by one...
        '''
        self.feasibility()
        self.populate_Q()
        self.optimize()
        self.optimize_transition()

    def policy(self, x=[0], input_method='index', output_method='best_input'):
        '''
        policy(x, input_method = 'index', output_method = 'best_input') returns the policy

        input_method = 'index'/'value'/'number'; currently, only supports index input
        output_method = 'best_input'/'action_number'/'action_index'/'action_value'
        :param x: by default is [0], when input_method == 'index', x should be the index in state_space.
        :param output_method: 'best_input' only works if you have done self.optimization_transition().
        'action_index/number/value' returns the best action.

        '''
        s_number = self.state_space.index_to_number(x)
        if output_method == 'best_input':
            return self.ωn.T[s_number]
        elif output_method == 'action_number':
            return self.ρ[s_number]
        elif output_method == 'action_index':
            return self.action_space.number_to_index(self.ρ[s_number])
        elif output_method == 'action_value':
            return self.action_space.number_to_value(self.ρ[s_number])

    def analyze_choice(self, test_func):
        '''
        analyze_choice(test_func) returns the subspace in state space, where
        test_func(**state_space, **action_space, **par) returns True.

        STILL WORKING
        '''
        return 0

    # Newton's algorithm for solving nonlinear system equations.


def iter_newton(X, function, imax=100, tol=1e-3, pace=1e-3, value_tol=1e-2,
                method='relative'):
    '''
    Newton Method Solving Simultaneous functions. (not very precise)
    iter_newton(X,function,imax = 100,tol = 1e-2, pace = 5e-2, value_tol = 1e-2,
                method = 'relative')
    Input: X: initial guess,
        function: a function f(X)
        imax: maximum iterations.
        tol: tolerance for changes of dX. If np.linalg.norm(dX)<tol, the iteration converges
        pace: pace to solve Jacobian matrix, using jaco(X, function, fX, pace)
        method: 'relative' the iteration breaks if a local minimum is achieved,
            or,before the sequences diverge.
            'function value' Instead, the iteration breaks if |f(X)|<value_tol
            'others' the traditional Newton algorithm is used (breaks when dX is close
            to 0.)
    Output: (X, Y) X is the solution, Y is the evaluation, which should be close to 0!

    '''
    X_history = []
    Y_history = []
    for i in range(int(imax)):
        # record a new X..
        X_history.append(np.array(X))
        print('=====Iteration=====', i)
        print('compute function value and jacobian matrix')
        Y = function(X)

        if method == 'function value':
            if max(np.abs(Y)) < value_tol:
                print('value_tol reached.')
                return (X, Y)
        print('examine X_history ')
        print(X_history)
        if i >= 1 and method == 'relative':
            if np.dot(Y, Y) > np.dot(last_Y, last_Y):
                print('Converging somehow fails. The function might be too discontinuous.')
                print('local minimum instead achieved. Return x value:')
                print(i)
                print(i - 1)
                print(X_history)
                res_X = X_history[i - 1]
                print(res_X)
                print('and Y value')
                print(Y_history[i - 1])

                return (res_X, Y_history[i - 1])

        J = jaco(X, function, fX=Y, pace=pace)  # calculate jacobian J = df(X)/dY(X)
        print(J)

        dX = np.linalg.solve(J, Y)  # solve for increment from JdX = Y

        X -= dX  # step X by dX

        Y_history.append(np.array(Y))
        if np.linalg.norm(dX) < tol:  # break if converged
            print('converged.')
            print('function value:')
            print(Y)
            return (X, Y)
        last_Y = Y


def jaco(X, function, fX, pace=1e-3):
    nx = np.array(X).size
    δf_jaco = []
    for j in range(nx):
        δX = np.zeros(nx)
        δX[j] = pace  # Definition of pace, that is, δX when only j changes
        δf = (function(X + δX) - fX) / (δX[j])
        δf_jaco.append(δf)
    δf_jaco = np.array(δf_jaco).T
    return δf_jaco


def print_dict(d):
    print(('Key   ', 'Value   '))
    for k, v in d.items():
        print((k, v))


def evaluate_onspace(func, x_space):
    '''
    DELETED IN THE FUTURE, REPLACED BY space.imap and anonymous_space.imap
    '''
    input_list = x_space('value')
    res = list(map(func, input_list))
    return res


def solve_mc_stationary(A):
    """ x = xA where x is the answer
    x - xA = 0
    x( I - A ) = 0 and sum(x) = 1
    """
    n = A.shape[0]
    a = np.eye(n) - A
    a = np.vstack((a.T, np.ones(n)))
    b = np.matrix([0] * n + [1]).T
    return np.array(np.linalg.lstsq(a, b)[0])


def model_par_diff(model, base_par, examined_pars, changed_par_values,
                   interested_variables, logging=False, output_property = 'diff',
                   **kwargs):
    """

    Parameters
    ----------
    model : function
        model(par, **kwargs) returns a dict like {'':, ...}.
    base_par : dict
        baseline parameters.
    examined_pars : list
        a list of parameters to change. e.g. ['r', 'r', 'n', 'g'] of which you vary the values.
    changed_par_values : list
        a list of parameter values corresponding to examine_pars. e.g. [3,3,5,np.linspace(1,2,3)]
    interested_variables : list
        a list of variables in returning dictionary of which you want to compose the table.
    logging : bool
        True if a distinct entry named 'logging' is recorded in the returning table. Notice that if Ture, model()
        must incorporate an entry 'logging' in the returning dictionary.
    output_property : str
        'diff' or 'return'. Use 'return' if you want the talbe to record the model returning variables,
        Use 'diff' if you want the table to record the variable difference to the baseline model.
    kwargs : dict
        other inputs for the table

    Returns
    -------
        pd.DataFrame
    """
    
    origional_par_values = list(base_par[j] for j in examined_pars)
    print('computing the origional model')
    market0 = model(base_par, **kwargs)
    market = []

    for i in range(len(examined_pars)):
        print('analyzing variables...')
        print(examined_pars[i])
        temp_par = base_par.copy()
        temp_par[examined_pars[i]] = changed_par_values[i]
        market_temp = model(temp_par, **kwargs)
        market_toappend = {}
        for j in list(market0.keys()):
            market_toappend[j] = [market_temp[j], market0[j]]
        market_toappend['issue'] = [examined_pars[i], 'changed to', changed_par_values[i], 'origional is',
                                    origional_par_values[i]]
        if logging:
            market_toappend['logging'] = market_temp['logging']
        market.append(market_toappend.copy())  # append the resulting dictionary

    def market_analyze(market, market0, interested_market_variables):
        '''
        Translating the list of aggregate markets into a pd.DataFrame object..
        '''
        res_dict = {}  # collect all results in this single dictionary, used for a DataFrame.

        # Initialize the res_dict
        for var_name in interested_market_variables:
            res_dict[var_name] = []
        res_dict['parameters'] = []
        if output_property == "diff":
            res_dict['parameter difference'] = []
        elif output_property == "return":
            res_dict['parameter value'] = []
        res_dict['parameter original'] = []
        if logging:
            res_dict['logging'] = []  #create entry of logging if True

        for i in range(len(market)):
            # market is a list of market profiles in dictionary.
            # in each loop, I compute the market changes..
            for var_name in interested_market_variables:
                if output_property == "diff":
                    res_dict[var_name].append(market[i][var_name][0] - market[i][var_name][1])
                elif output_property == "return":
                    res_dict[var_name].append(market[i][var_name][0])
            res_dict['parameters'].append(market[i]['issue'][0])
            if output_property == "diff":
                res_dict['parameter difference'].append(market[i]['issue'][2] - market[i]['issue'][4])
            elif output_property == "return":
                res_dict['parameter value'].append(market[i]['issue'][2])
            res_dict['parameter original'].append(market[i]['issue'][4])
            if logging:
                res_dict['logging'].append(market[i]['logging'])

        for var_name in interested_market_variables:
            res_dict[var_name].append(market0[var_name])
        res_dict['parameters'].append('Baseline Model')
        if output_property == "diff":
            res_dict['parameter difference'].append(np.nan)
        elif output_property == "return":
            res_dict['parameter value'].append(np.nan)
        res_dict['parameter original'].append(np.nan)
        if logging:
            res_dict['logging'].append(market0['logging'])

        return res_dict

    market_diff = market_analyze(market, market0, interested_variables)
    market_diff = pd.DataFrame(data=market_diff)  # turn dictionary into df
    ##
    return market_diff


def model_par(model, base_par, examined_pars, changed_par_values,
                   interested_variables, logging=False, output_property='diff', parallel = 8,
                   **kwargs):
    """
    model_par is model_par_diff + multiprocessing.
    Parameters
    ----------
    model : function
        model(par, **kwargs) returns a dict like {'':, ...}.
    base_par : dict
        baseline parameters.
    examined_pars : list
        a list of parameters to change. e.g. ['r', 'r', 'n', 'g'] of which you vary the values.
    changed_par_values : list
        a list of parameter values corresponding to examine_pars. e.g. [3,3,5,np.linspace(1,2,3)]
    interested_variables : list
        a list of variables in returning dictionary of which you want to compose the table.
    logging : bool
        True if a distinct entry named 'logging' is recorded in the returning table. Notice that if Ture, model()
        must incorporate an entry 'logging' in the returning dictionary.
    output_property : str
        'diff' or 'return'. Use 'return' if you want the talbe to record the model returning variables,
        Use 'diff' if you want the table to record the variable difference to the baseline model.
    kwargs : dict
        other inputs for the table

    Returns
    -------
        pd.DataFrame
    """

    origional_par_values = list(base_par[j] for j in examined_pars)

    def worker(i):
        if i == len(examined_pars):
            market0 = model(base_par, **kwargs)
            return market0
        print('analyzing variables...')
        print(examined_pars[i])
        temp_par = base_par.copy()
        temp_par[examined_pars[i]] = changed_par_values[i]
        market_temp = model(temp_par, **kwargs)
        market_toappend = {}
        for j in list(interested_variables):
            market_toappend[j] = market_temp[j]
            market_toappend['issue'] = [examined_pars[i], 'changed to', changed_par_values[i], 'origional is',
                                    origional_par_values[i]]
        if logging:
            market_toappend['logging'] = market_temp['logging']
        return market_toappend

    pool = ProcessPool(nodes = parallel)
    market_all = list(pool.map(worker, list(range(len(examined_pars)+1))))

    market = market_all[0:len(examined_pars)]
    market0 = market_all[len(examined_pars)]

    ''' # Original loop
    for i in range(len(examined_pars)):
        print('analyzing variables...')
        print(examined_pars[i])
        temp_par = base_par.copy()
        temp_par[examined_pars[i]] = changed_par_values[i]
        market_temp = model(temp_par, **kwargs)
        market_toappend = {}
        for j in list(market0.keys()):
            market_toappend[j] = [market_temp[j], market0[j]]
        market_toappend['issue'] = [examined_pars[i], 'changed to', changed_par_values[i], 'origional is',
                                    origional_par_values[i]]
        if logging:
            market_toappend['logging'] = market_temp['logging']
        market.append(market_toappend.copy())  # append the resulting dictionary
    '''
    def market_analyze(market, market0, interested_market_variables):
        '''
        Translating the list of aggregate markets into a pd.DataFrame object..
        '''
        res_dict = {}  # collect all results in this single dictionary, used for a DataFrame.

        # Initialize the res_dict
        for var_name in interested_market_variables:
            res_dict[var_name] = []
        res_dict['parameters'] = []
        if output_property == "diff":
            res_dict['parameter difference'] = []
        elif output_property == "return":
            res_dict['parameter value'] = []
        res_dict['parameter original'] = []
        if logging:
            res_dict['logging'] = []  # create entry of logging if True

        for i in range(len(market)):
            # market is a list of market profiles in dictionary.
            # in each loop, I compute the market changes..
            for var_name in interested_market_variables:
                if output_property == "diff":
                    res_dict[var_name].append(market[i][var_name] - market0[i][var_name])
                elif output_property == "return":
                    res_dict[var_name].append(market[i][var_name])
            res_dict['parameters'].append(market[i]['issue'][0])
            if output_property == "diff":
                res_dict['parameter difference'].append(market[i]['issue'][2] - market[i]['issue'][4])
            elif output_property == "return":
                res_dict['parameter value'].append(market[i]['issue'][2])
            res_dict['parameter original'].append(market[i]['issue'][4])
            if logging:
                res_dict['logging'].append(market[i]['logging'])

        for var_name in interested_market_variables:
            res_dict[var_name].append(market0[var_name])
        res_dict['parameters'].append('Baseline Model')
        if output_property == "diff":
            res_dict['parameter difference'].append(np.nan)
        elif output_property == "return":
            res_dict['parameter value'].append(np.nan)
        res_dict['parameter original'].append(np.nan)
        if logging:
            res_dict['logging'].append(market0['logging'])

        return res_dict

    market_diff = market_analyze(market, market0, interested_variables)
    market_diff = pd.DataFrame(data=market_diff)  # turn dictionary into df
    ##
    return market_diff


def before_foo(par, time):
    return par


def after_foo(result_dict, par, time):
    '''after_foo returns the updated next-period environmental par
    Input:
        result_dict: a dictionary
        par: a dictionary of environmental parameters
        time: what time is it?
    output:
        updated_par: a dictionary of updated environmental parameters.
    '''
    return par


def recursive_model(model, init_par={}, step=3, before_func=before_foo,
                    after_func=after_foo, **kwargs):
    """
    recursive_model returns a list of model result dictionaries,
    if the model is run recursively, with each time the resulting dictionary
    changes the environmental parameters.
    \n
    initial_par -> before_func(initial_par, time) -> model(..., **kwargs) -> after_func(result_dict,
    par, time) -> ... before_func(...,time)...


    Parameters
    ----------
    model : function
        a function like model(par, **kwargs) that returns resulting dictionary.
    init_par : dict
        initial parameters of the model
    step : int
        how long it takes?
    before_func : function
        before_func(par, time) returns a new par (used when you want to change
                                a parameter at a certain time)
    after_func : function
        after_func(result_dict, par, time) returns a new par, used when
            you want the returns of the model feeds the environment.
    kwargs : dict
        other constant inputs into the model

    Returns
    -------
    dict
        {'par_seq': par_sequence, 'res_seq': res_sequence}
    """
    par = init_par.copy()
    par_sequence = []
    res_sequence = []

    for i in range(step):
        # 1. shock to parameters
        par = before_func(par, i)
        par_sequence.append(par.copy())  # append the par used in computing this period
        # problem to the returning list

        # 2. compute the model
        res = model(par, **kwargs)
        res_sequence.append(res.copy())  # attach the result to returning list

        # 3. perform the influnce of the model to parameters.
        par = after_func(res, par, i)

    result = {'par_seq': par_sequence, 'res_seq': res_sequence}
    return result

def list_dict_to_DataFrame(list_dict, var_name):
    """
    distill entries from each dict, and collect them into a pandas.DataFrame.
    \n
    Parameters
    ----------
    list_dict : list
        list of dictionaries.
    var_name : list
        list of str.

    Returns
    -------
    pandas.DataFrame

    """
    res = {}
    for var in var_name:
        res[var] = [dic[var] for dic in list_dict]
    return pd.DataFrame(data = res)

if __name__ == '__main__':
    '''
    The following code showcases a simple static optimization problem.
    '''


    def f_vec(x1, x2, x3, y1, y2):
        return (np.sin(y1) + (y1 - y2 - x3) ** 2 + x1 * x2 / y1)


    f_vec = np.vectorize(f_vec)



    def constrain(x1, x2, x3, y1, y2):
        return y1 + y2 > x3 / x1


    constrain = np.vectorize(constrain)

    xo = optim("max", state={"x1": np.linspace(1, 3, 5), "x2": np.linspace(3, 6, 8),
                             "x3": np.linspace(6, 10, 3)},
               action={"y1": np.linspace(1, 3, 2), "y2": np.linspace(3, 6, 3)},
               constrain=constrain,
               u=f_vec, parallel=1)
    qe.util.tic()
    xo.feasibility()
    qe.util.toc()

    qe.util.tic()
    xo.optimize()
    qe.util.toc()
