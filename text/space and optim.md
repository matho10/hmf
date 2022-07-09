# Space and optim: Packages for modelling numerics and optimization

## Defining computational regions

### Important concepts
Suppose we have a plain like $E\subset\mathbb{R}^{2}$. Subset $E$ is defined by a *constrain function* $g(x,y)$ with $E={(x,y):g(x,y)=True}$. We want to do a mesh plot of $f(x,y)$. Usually, we need to start with defining proper `meshgrids` in python and select the points that fall into set $E$. With the help of `space` and `optim`, this goal is achieved quite easily.

First, let us set out the following concepts in an `space` object:

1. `set`: the `np.array` object containing all points along each axis. In this case, let us say `x=np.linspace(0,10,100)` and `y=np.linspace(0,10,100)`. 
2. `name`: `x` and `y`
3. `grid_dict` a dictionary contains all grids along each axis.
4. concept of number, index, and value: number are natural numbers in all numbered possible combinations of x and y. These numbers range from 0 to (the number of all possible combinations of x and y) -1. Index is a list of numbers pointing to specific locations in each set. Value are the exact values of (x,y). For our case, number 1  = index (0, 1) = value (0, 0.1). 

By defining `s = space(x = np.linspace(0,10,100), y=np.linspace(0,10,100))` we create a named space object, which contains following properties:

`name`, `set`, `grid_dict`, `dict`, `shape`, `N`, `number`, `number_matrix`, 

and usable functions:

`index_to_number, number_to_index, index_to_value, number_to_value, __call__, map, imap`

### use optim to do graph







