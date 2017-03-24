from simtk.openmm import XmlSerializer
from benchmark.integrators.langevin import LangevinSplittingIntegrator
from benchmark import DATA_PATH
import os

schemes = ["V R O R V", "O R V R O", "R V O V R",
           "O V R V O", "R R V O V R R",
           "O V R R R R V O", "V R R O R R V",
           "V R R R O R R R V"]

step_type_dict = {
    0 : "{target} <- {expr}",
    1: "{target} <- {expr}",
    2: "{target} <- sum({expr})",
    3: "constrain positions",
    4: "constrain velocities",
    5: "allow forces to update the context state",
    6: "if:",
    7: "while:",
    8: "end"
}

for scheme in schemes:
    print(scheme)
    integrator = LangevinSplittingIntegrator(scheme)

    # Export integrator to XML
    with open(os.path.join(DATA_PATH, "serialized_integrators/{}.xml".format(scheme), "w")) as f:
        f.writelines(XmlSerializer.serialize(integrator))

    # Also pretty-print (using `step_type_dict`)
    readable_lines = []
    for i in range(integrator.getNumComputations()):
        step_type, target, expr = integrator.getComputationStep(i)
        readable_lines.append(step_type_dict[step_type].format(target=target, expr=expr) + "\n")
    print(readable_lines)

    # Save pretty-printed results to .txt
    with open(os.path.join(DATA_PATH, "serialized_integrators/{}.txt".format(scheme), "w")) as f:
        f.writelines(readable_lines)
