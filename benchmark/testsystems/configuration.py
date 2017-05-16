import simtk.openmm as mm

from openmmtools.testsystems import HarmonicOscillator


def check_platform(platform):
    """Check whether we can construct a simulation using this platform"""
    try:
        integrator = mm.VerletIntegrator(1.0)
        testsystem = HarmonicOscillator()
        context = mm.Context(testsystem.system, integrator, platform)
        del context, testsystem, integrator
    except Exception:
        print('Desired platform not supported')


def configure_platform(platform_name='Reference', fallback_platform_name='CPU'):
    """
    Retrieve the requested platform with platform-appropriate precision settings.

    platform_name : str, optional, default='Reference'
       The requested platform name
    fallback_platform_name : str, optional, default='CPU'
       If the requested platform cannot be provided, the fallback platform will be provided.

    Returns
    -------
    platform : simtk.openmm.Platform
       The requested platform with precision configured appropriately,
       or the fallback platform if this is not available.

    """
    fallback_platform = mm.Platform.getPlatformByName(fallback_platform_name)
    try:
        if platform_name.upper() == 'Reference'.upper():
            platform = mm.Platform.getPlatformByName('Reference')
        elif platform_name.upper() == "CPU":
            platform = mm.Platform.getPlatformByName("CPU")
        elif platform_name.upper() == 'OpenCL'.upper():
            platform = mm.Platform.getPlatformByName('OpenCL')
            platform.setPropertyDefaultValue('OpenCLPrecision', 'mixed')
        elif platform_name.upper() == 'CUDA'.upper():
            platform = mm.Platform.getPlatformByName('CUDA')
            platform.setPropertyDefaultValue('CUDAPrecision', 'mixed')
        else:
            raise (ValueError("Invalid platform name"))

        check_platform(platform)

    except:
        print(
        "Warning: Returning {} platform instead of requested platform {}".format(fallback_platform_name, platform_name))
        platform = fallback_platform

    return platform
