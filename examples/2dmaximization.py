'''
A simple maximization problem:
max x*sin(x) * cos( a* y) s.t. x+y <=10-a, x>=0, y>=0, and a is a parameter (a<10).


'''


from hmf3 import space, optim
import numpy as np
import scipy as sp


def u(z, x, y, a):
    return x**2 - (y-a) **2

def constrain(z, x, y, a):
    return (x**2 + y**2<=4)


if __name__=="__main__":
    s = optim(state = {'z':np.array([1])}, action = {'x': np.arange(0,5,0.1), 'y': np.arange(0,5, 0.1)},
              constrain = constrain, u = u,
              par = {'a': 1})
    s.feasibility()
    s.optimize()
    print(s.v)
    print(s.policy(output_method='action_value'))

