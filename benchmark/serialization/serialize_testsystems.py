# also serialize all the testsystems
from simtk.openmm import XmlSerializer
from code.testsystems import system_params

for name in system_params.keys():
    print(name)
    for constrained in [True, False]:
        top, sys, pos = system_params[name]["loader"](constrained)
        c = "constrained"
        if not constrained: c = "unconstrained"
        with open("serialized_testsystems/{}_{}.xml".format(name, c), "w") as f:
            f.writelines(XmlSerializer.serialize(sys))

    ## also pretty-print
    #readable_lines = []
    #for i in range(integrator.getNumComputations()):
    #    step_type, target, expr = integrator.getComputationStep(i)
    #    readable_lines.append(step_type_dict[step_type].format(target=target, expr=expr) + "\n")
    #print(readable_lines)

    #with open("serialized_integrators/readable_{}.txt".format(scheme), "w") as f:
    #    f.writelines(readable_lines)
