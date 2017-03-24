# also serialize all the testsystems
from simtk.openmm import XmlSerializer
from benchmark.testsystems import system_params

for name in system_params.keys():
    print(name)
    for constrained in [True, False]:
        top, sys, pos = system_params[name]["loader"](constrained)
        c = "constrained"
        if not constrained: c = "unconstrained"
        with open("serialized_testsystems/{}_{}.xml".format(name, c), "w") as f:
            f.writelines(XmlSerializer.serialize(sys))
