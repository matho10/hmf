# Heterogeneous Modeling Functionality (not finished)

Heterogeneous modeling functionality is a repository for python functions that enable one to model complicated dynamic programming problems for a pool of heterogeneous agents, study the their aggregate behavior, and compute the equilibrium if there exists one. 

These applications mostly arise in the field of macroeconomic, finance and computer science.

The core of the problem is the formalization into a state space model from an intuitive Bellman equation. 
$$
V(x)=u(x,c)+\beta \int_{x'\in G(x,c)} f(x'|x,c)V(x')dx',
$$
where $f(x'|x,c)$ is called kernel function, a conditional probability density function given current state $x$ and choice $c$. This is a standard optimal control problem and the key to understand its solvability is the concavity of the functional involved here. For further mathematical studies, see []..

When putting equation (1) into real actions, we face a great degree of difficulties. First, it is not straightforward to translate a continuous space into numerics that a computer usually understands. This difficulty is addressed by `hmf3.space` which is efficient in building, indexing and reshaping the joint space of states and actions. Second, it is not easy to formulate 




