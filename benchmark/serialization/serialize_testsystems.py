# also serialize all the testsystems
from simtk.openmm import XmlSerializer
from benchmark.testsystems import system_loaders

for name in system_loaders.keys():
    print(name)
    for constrained in [True, False]:
        top, sys, pos = system_loaders[name](constrained)
        c = "constrained"
        if not constrained:
            c = "unconstrained"
        with open("{}_{}.xml".format(name, c), "wt") as f:
            f.writelines(XmlSerializer.serialize(sys))
