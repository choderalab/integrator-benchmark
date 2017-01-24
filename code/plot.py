from integrator_benchmark import plot

from cPickle import load
name = "constrained_randomized"
f = open(name + "_results.pkl", "r")
results = load(f)
f.close()
plot(results, name)