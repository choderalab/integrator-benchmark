import sys

if __name__ == "__main__":
    sys_id = int(sys.argv[1])

    # DHFR
    if sys_id == 1:
        from benchmark.testsystems import dhfr_constrained
        system = dhfr_constrained

    elif sys_id == 2:
        from benchmark.testsystems import dhfr_unconstrained
        system = dhfr_unconstrained

    # WaterBox
    elif sys_id == 3:
        from benchmark.testsystems import waterbox_constrained
        system = waterbox_constrained

    elif sys_id == 4:
        from benchmark.testsystems import flexible_waterbox
        system = flexible_waterbox

    # T4 lysozyme
    elif sys_id == 5:
        from benchmark.testsystems import t4_constrained
        system = t4_constrained

    elif sys_id == 6:
        from benchmark.testsystems import t4_unconstrained
        system = t4_unconstrained

    # alanine dipeptide
    elif sys_id == 7:
        from benchmark.testsystems import alanine_constrained
        system = alanine_constrained

    elif sys_id == 8:
        from benchmark.testsystems import alanine_unconstrained
        system = alanine_unconstrained

    else:
        raise(Exception("Invalid array-job index!!"))

    system.sample_x_from_equilibrium()
