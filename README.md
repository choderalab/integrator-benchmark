[![Build Status](https://travis-ci.org/choderalab/integrator-benchmark.svg?branch=master)](https://travis-ci.org/choderalab/integrator-benchmark?branch=master)


# Not all Langevin integrators are equal

Enumerating and evaluating numerical schemes for Langevin dynamics

## Molecular dynamics requires methods for simulating continuous equations in discrete steps
A widely-used approach to investigating equilibrium properties is to simulate Langevin dynamics.
Langevin dynamics is a system of stochastic differential equations defined on a state space of configurations $\mathbf{x}$ and velocities $\mathbf{v}$.

To simulate those equations on a computer, we need to provide explicit instructions for advancing the state of the system $(\mathbf{x},\mathbf{v})$` by very small time increments.

Here, we will consider the family of methods that can be derived by splitting the Langevin system into a sum of three simpler systems, labeled `O`, `R`, and `V`, and approximately propagating each of those simpler systems for small increments of time.

(TODO: Add LaTeX-rendered Langevin system with underbraces around `O`, `R`, `V` components, using `readme2tex`)

We will refer to a numerical scheme by its encoding string and finite timestep $\Delta t$, i.e. `OVRVO` means simulate the `O` component for an increment of $\Delta t/2$`, then the `V` component for an increment of $\Delta t/2$, then the `R` component for an increment of $\Delta t$, then the `V` component for an increment of $\Delta t/2$, the `O` component for an increment of $\Delta t/2$. This approximately propagates the entire system for a total time increment of $\Delta t$.

## This introduces error that can be sensitive to details
Subtly different numerical schemes for the same continuous equations can have drastically different behavior at finite timesteps.
As a prototypical example, consider the difference between the behavior of schemes `VRORV` and `OVRVO` on a 1D quartic system:

![quartic_eq_joint_dist_array_w_x_marginals](https://cloud.githubusercontent.com/assets/5759036/25289560/147862fa-2698-11e7-8f95-9b463953f2de.jpg)

($\rho$ is the density sampled by the numerical scheme, and $\pi$ is the exact target density.
Each column illustrates an increasing finite timestep $\Delta t$, below the "stability threshold."
Rows 2 and 4 illustrate error in the sampled joint distribution, and rows 1 and 3 illustrate error in the sampled $\mathbf{x}$-marginal distribution.)

The two schemes have the same computational cost per iteration, and introduce nearly identical levels of error into the sampled joint $(\mathbf{x}, \mathbf{v})$ distribution -- but one of these methods introduces nearly 100x more error in the $\mathbf{x}$ marginal than the other at large timesteps!

### Velocity verlet with velocity randomization (`OVRVO`)
For concreteness, here's a toy Python implementation of the inner-loop of the scheme denoted `OVRVO`:
```python
# O step (dt/2)
v = (a * v) + b * (velocity_scale * randn())

# V step (dt/2)
v = v + ((dt / 2.0) * force(x) / m)

# R step (dt)
x = x + (dt * v)

# V step (dt/2)
v = v + ((dt / 2.0) * force(x) / m)

# O step (dt/2)
v = (a * v) + b * (velocity_scale * randn())
```

where
```python
import numpy as np
from numpy.random import randn

velocity_scale = np.sqrt(1.0 / (beta * m))
# a, b are constants used in exact distributional solves of the O component for (dt/2)
a = np.exp(-gamma * (dt / 2.0))
b = np.sqrt(1 - np.exp(-2 * gamma * (dt / 2.0)))
```


### BAOAB (`VRORV`)
And here's code for the `VRORV` inner-loop:
```python
# V step (dt/2)
v = v + ((dt / 2.0) * force(x) / m)

# R step (dt/2)
x = x + ((dt/2.0) * v)

# O step (dt)
v = (a * v) + b * (velocity_scale * randn())

# R step (dt/2)
x = x + ((dt/2.0) * v)

# V step (dt/2)
v = v + ((dt / 2.0) * force(x) / m)
```

where
```python
# a, b are constants used in exact distributional solves of the O component for dt

a = np.exp(-gamma * dt)
b = np.sqrt(1 - np.exp(-2 * gamma * dt))
```

# Systematically enumerating numerical schemes and measuring their error
In this repository, we enumerate numerical schemes for Langevin dynamics by associating strings over the alphabet `{O, R, V}` with explicit numerical methods using OpenMM `CustomIntegrator`s, and providing schemes for approximating the error introduced by that method in the sampled distribution over `(x,v)` jointly or `x` alone using nonequilibrium work theorems.
We further investigate the effects of modifying the mass matrix (aka "hydrogen mass repartitioning") and/or evaluating subsets of the forces per substep (aka "multi-timestep" methods, obtained by expanding the alphabet to `{O, R, V0, V1, ..., V32}`).

(TODO: Details on nonequilibrium error measurement scheme.)

## Relation to prior work
We did not introduce the concept of splitting the Langevin system into these three "building blocks" -- this decomposition is developed lucidly in Chapter 7 of [[Leimkuhler and Matthews, 2015]](http://www.springer.com/us/book/9783319163741). We also did not discover the `VRORV` integrator -- Leimkuhler and Matthews have studied the favorable properties of particular integrator `VRORV` ("`BAOAB`" in their notation) in great detail.
Here, we have (1) "compiled" these strings into efficient `CustomIntegrators` in OpenMM, (2) provided a more "universal" method for measuring the sampling error introduced by any member of this family (approximating the KL divergence directly, rather than monitoring error in a system-specific choice of low-dimensional observable), (3) considered a slightly expanded alphabet.

# References
(TODO: Add references from paper!)