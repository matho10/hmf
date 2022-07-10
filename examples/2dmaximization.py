'''
A simple maximization problem:
max x*sin(x) * cos( a* y) s.t. x+y <=10-a, x>=0, y>=0, and a is a parameter (a<10).


'''


from hmf3 import space, optim
import numpy as np
import quantecon as qe


def u(z, x, y, a):
    return x**2 - (y-a) **2

def constrain(z, x, y, a):
    return (x**2 + y**2<=40)


if __name__=="__main__":
    s = optim(state = {'z':np.array([1])}, action = {'x': np.arange(0,15,0.5), 'y': np.arange(0,15, 0.5)},
              constrain = constrain, u = u,
              par = {'a': 1})
    qe.util.tic()
    print('feasibility takes time')
    s.feasibility()
    qe.util.toc()

    qe.util.tic()
    print('optimization takes time')
    s.optimize()
    qe.util.toc()

    print('value on states are')
    print(s.v)
    print('policy function')
    print(s.policy(output_method='action_value'))


