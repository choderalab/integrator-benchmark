
import simtk.openmm as mm


def configure_platform(platform_name='Reference'):
    """Set precision, etc..."""
    if platform_name.upper() == 'Reference'.upper():
        platform = mm.Platform.getPlatformByName('Reference')
    elif platform_name.upper() == 'OpenCL'.upper():
        platform = mm.Platform.getPlatformByName('OpenCL')
        platform.setPropertyDefaultValue('OpenCLPrecision', 'mixed')
    elif platform_name.upper() == 'CUDA'.upper():
        platform = mm.Platform.getPlatformByName('CUDA')
        platform.setPropertyDefaultValue('CUDAPrecision', 'double')
    else:
        raise(ValueError("Invalid platform name"))
    return platform