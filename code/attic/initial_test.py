from .integrators import LangevinSplittingIntegrator

if __name__=="__main__":
    #test_system = "alanine"
    test_system = "waterbox"

    def get_test_system(constrained=True):
        if test_system == 'alanine':
            from openmmtools.testsystems import AlanineDipeptideVacuum
            if constrained:
                testsystem = AlanineDipeptideVacuum()
            else:
                testsystem = AlanineDipeptideVacuum(constraints=None)
        elif test_system == 'waterbox':
            from openmmtools.testsystems import WaterBox
            testsystem = WaterBox(constrained=constrained)

        return testsystem

    testsystem = get_test_system()

    (system, positions) = testsystem.system, testsystem.positions
    print('Dimensionality: {}'.format(len(positions) * 3))

    from simtk import openmm, unit
    temperature = 300 * unit.kelvin

    def test(starting_positions,
             scheme='RVOVR',
             timestep=1.0*unit.femtoseconds,
             constrained=True,
             n_steps=5000
             ):
        testsystem = get_test_system(constrained)
        (system, positions) = testsystem.system, testsystem.positions
        integrator = LangevinSplittingIntegrator(scheme,
                                                 temperature=temperature,
                                                 timestep=timestep,
                                                 )

        #platform = mm.Platform.getPlatformByName('Reference')
        #simulation = app.Simulation(testsystem.topology, system, integrator, platform)

        platform = mm.Platform.getPlatformByName('OpenCL')
        platform.setPropertyDefaultValue('OpenCLPrecision', 'mixed')
        simulation = app.Simulation(testsystem.topology, system, integrator, platform)

        context = simulation.context
        context.setPositions(starting_positions)
        context.setVelocitiesToTemperature(temperature)
        energy_unit = unit.kilojoule_per_mole

        def total_energy():
            state = context.getState(getEnergy=True)
            ke = state.getKineticEnergy()
            pe = state.getPotentialEnergy()
            return (pe + ke).value_in_unit(energy_unit)

        heats = [0]
        shadow_works = [0]
        energies = [total_energy()]

        for i in range(n_steps):
            integrator.step(1)
            energies.append(total_energy())
            heats.append(integrator.getGlobalVariableByName('heat'))
            shadow_works.append((energies[-1] - energies[0]) - heats[-1])
        return context, heats, shadow_works

    # non-equilibrated initial conditions
    n_replicates = 2
    plt.figure()
    for _ in range(n_replicates):
        context, heats, shadow_works = test(positions)
        plt.plot(shadow_works, c='blue')
    plt.xlabel('# steps')
    plt.ylabel('W_shad')
    plt.title('Shadow work (non-equilibrated initial conditions)')
    plt.savefig('shadow_work_plot.jpg')
    plt.close()


    # equilibrated initial conditions
    plt.figure()
    equilibrated = context.getState(getPositions=True).getPositions()
    for _ in range(n_replicates):
        context, heats, shadow_works = test(equilibrated)
        plt.plot(shadow_works, c='blue')

    plt.xlabel('# steps')
    plt.ylabel('W_shad')
    plt.title('Shadow work (equilibrated initial conditions)')
    plt.savefig('shadow_work_plot_after_equilibration.jpg')
    plt.close()

    # many choices now
    def do_comparison(constrained=True, transient=1000):

        colors = {'RVOVR': 'blue', 'VRORV':'red', 'OVRVO':'green'}
        plt.figure()

        timesteps = numpy.array([0.5, 1, 2, 2.5])
        if constrained: timesteps = timesteps * 2

        for scheme in colors.keys():
            for timestep in timesteps:
                label = '{} ({} fs)'.format(scheme, timestep)

                context, heats, shadow_works = test(positions, scheme, timestep * unit.femtoseconds, constrained)
                equilibrated = context.getState(getPositions=True).getPositions()

                context, heats, shadow_works = test(equilibrated,
                                             scheme=scheme,
                                             timestep=timestep*unit.femtoseconds,
                                             constrained=constrained
                                             )
                print('{}: {:.3f}'.format(label, shadow_works[-1]))
                plt.plot(numpy.array(shadow_works[transient:]) - shadow_works[transient], c=colors[scheme], label=label, linewidth=timestep)
        plt.legend(loc=(1,0))
        plt.tight_layout()
        plt.xlabel('# steps')
        plt.ylabel('W_shad')
        name = 'Integrator comparison'
        if constrained: name = name + ' (constrained)'
        else: name = name + ' (unconstrained)'
        plt.title(name)

    for constrained in [True, False]:
        print('Constraints: {}'.format(constrained))
        do_comparison(constrained)
        plt.savefig('shadow_work_comparison_{}.jpg'.format(constrained), bbox_inches='tight')