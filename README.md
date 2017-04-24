[![Build Status](https://travis-ci.org/choderalab/integrator-benchmark.svg?branch=master)](https://travis-ci.org/choderalab/integrator-benchmark?branch=master)


# Not all Langevin integrators are equal

Enumerating and evaluating numerical schemes for Langevin dynamics

## Molecular dynamics requires methods for simulating continuous equations in discrete steps
A widely-used approach to investigating equilibrium properties is to simulate Langevin dynamics.
Langevin dynamics is a system of stochastic differential equations defined on a state space of configurations $\mathbf{x}$ and velocities $\mathbf{v}$.

To simulate those equations on a computer, we need to provide explicit instructions for advancing the state of the system $(\mathbf{x},\mathbf{v})$` by very small time increments.

Here, we will consider the family of methods that can be derived by splitting the Langevin system into a sum of three simpler systems, labeled `O`, `R`, and `V`. We then define a numerical method by approximately propagating each of those simpler systems for small increments of time in a specified order.

(TODO: Add LaTeX-rendered Langevin system with underbraces around `O`, `R`, `V` components, using `readme2tex`)

We will refer to a numerical scheme by its encoding string. For example, `OVRVO` means: simulate the `O` component for a time increment of $\Delta t/2$, then the `V` component for $\Delta t/2$, then the `R` component for $\Delta t$, then the `V` component for $\Delta t/2$, and finally the `O` component for $\Delta t/2$. This approximately propagates the entire system for a total time increment of $\Delta t$.

## This introduces error that can be sensitive to details
Using subtly different numerical schemes for the same continuous equations can lead to drastically different behavior at finite timesteps.
As a prototypical example, consider the difference between the behavior of schemes `VRORV` and `OVRVO` on a 1D quartic system:

![quartic_eq_joint_dist_array_w_x_marginals](https://cloud.githubusercontent.com/assets/5759036/25289560/147862fa-2698-11e7-8f95-9b463953f2de.jpg)

($\rho$ is the density sampled by the numerical scheme, and $\pi$ is the exact target density.
Each column illustrates an increasing finite timestep $\Delta t$, below the "stability threshold."
Rows 2 and 4 illustrate error in the sampled joint distribution, and rows 1 and 3 illustrate error in the sampled $\mathbf{x}$-marginal distribution.)

The two schemes have the same computational cost per iteration, and introduce nearly identical levels of error into the sampled joint $(\mathbf{x}, \mathbf{v})$ distribution -- but one of these methods introduces about 100x more error in the $\mathbf{x}$ marginal than the other at large timesteps!

### Toy implementation
To illustrate the scheme, here is a toy Python implementation for each of the explicit updates:
```python
# Defined elsewhere: friction coefficient `gamma`, mass `m`

def propagate_R(x, v, h): 
    """Linear "drift" -- deterministic position update
    using current velocities"""
    return (x + (h * v), v)

def propagate_V(x, v, h):
    """Linear "kick" -- deterministic velocity update
    using current forces"""
    return (x, v + (h * force(x) / m))

def propagate_O(x, v, h):
    """Ornstein-Uhlenbeck -- stochastic velocity update
    using a ficticious "heat-bath""""
    a, b = exp(-gamma * h), sqrt(1 - exp(-2 * gamma * h))
    return (x, (a * v) + b * draw_maxwell_boltzmann_velocities())

propagate = {"O": propagate_O, "R": propagate_R, "V": propagate_V}
```
where `draw_maxwell_boltzmann_velocities()` draws an independent sample from the velocity distribution given by the masses and temperature.

Using the functions we just defined, here's how to implement the inner-loop of the scheme denoted `OVRVO`:
```python
x, v = propagate_O(x, v, dt / 2)
x, v = propagate_V(x, v, dt / 2)
x, v = propagate_R(x, v, dt)
x, v = propagate_V(x, v, dt / 2)
x, v = propagate_O(x, v, dt / 2)
```

And here's the `VRORV` inner-loop:
```python
x, v = propagate_V(x, v, dt / 2)
x, v = propagate_R(x, v, dt / 2)
x, v = propagate_O(x, v, dt)
x, v = propagate_R(x, v, dt / 2)
x, v = propagate_V(x, v, dt / 2)
```

As suggested from these examples, the generic recipe for turning a splitting string into a Langevin integrator is:

```python
def simulate_timestep(x, v, dt, splitting="OVRVO"):
    n_O = sum([substep == "O" for step in splitting])
    n_R = sum([substep == "R" for step in splitting])
    n_V = sum([substep == "V" for step in splitting])
    n = {"O": n_O, "R": n_R, "V": n_V}

    for substep in splitting:
        x, v = propagate[substep](x, v, dt / n[substep])
    return x, v
```

# Systematically enumerating numerical schemes and measuring their error
In this repository, we enumerate numerical schemes for Langevin dynamics by associating strings over the alphabet `{O, R, V}` with explicit numerical methods using [OpenMM `CustomIntegrator`](http://docs.openmm.org/7.1.0/userguide/theory.html#customintegrator)s. We provide schemes for approximating the error introduced by that method in the sampled distribution over $(\mathbf{x},\mathbf{v}) jointly or $\mathbf{x}$ alone using nonequilibrium work theorems.
We further investigate the effects of modifying the mass matrix (aka "hydrogen mass repartitioning") and/or evaluating subsets of the forces per substep (aka "multi-timestep" methods, obtained by expanding the alphabet to `{O, R, V0, V1, ..., V32}`).

(TODO: Details on nonequilibrium error measurement scheme.)

## Relation to prior work
We did not introduce the concept of splitting the Langevin system into these three "building blocks" -- this decomposition is developed lucidly in Chapter 7 of [[Leimkuhler and Matthews, 2015]](http://www.springer.com/us/book/9783319163741). We also did not discover the `VRORV` integrator -- Leimkuhler and Matthews have studied the favorable properties of particular integrator `VRORV` ("`BAOAB`" in their notation) in great detail.
Here, we have:
1. provided a method to translate these strings into efficient `CustomIntegrators` in OpenMM,
2. provided a uniform scheme for measuring the sampling error introduced by any member of this family of methods on any target density (approximating the KL divergence directly, rather than monitoring error in a system-specific choice of low-dimensional observable),
3. considered an expanded alphabet, encompassing many widely-used variants as special cases.

# References
(TODO: Add references from paper!)