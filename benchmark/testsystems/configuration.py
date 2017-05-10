
import simtk.openmm as mm

# Check if we support the OpenCL platform in mixed precision
_opencl_supports_mixed_precision = True
from openmmtools.testsystems import HarmonicOscillator
try:
    integrator = mm.VerletIntegrator(1.0)
    testsystem = HarmonicOscillator()
    platform = mm.Platform.getPlatformByName('OpenCL')
    properties = {'OpenCLPrecision' : 'mixed'}
    context = mm.Context(testsystem.system, integrator, platform, properties)
    del context, properties, testsystem, integrator
except Exception as e:
    print('OpenCL unavailable or does not support mixed precision')
    _opencl_supports_mixed_precision = False

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
    try:
        if platform_name.upper() == 'Reference'.upper():
            platform = mm.Platform.getPlatformByName('Reference')
        elif platform_name.upper() == "CPU":
            platform = mm.Platform.getPlatformByName("CPU")
        elif platform_name.upper() == 'OpenCL'.upper():
            if not _opencl_supports_mixed_precision:
                # Return CPU platform if OpenCL doesn't support mixed precision
                return mm.Platform.getPlatformByName(fallback_platform_name)
            platform = mm.Platform.getPlatformByName('OpenCL')
            platform.setPropertyDefaultValue('OpenCLPrecision', 'mixed')
        elif platform_name.upper() == 'CUDA'.upper():
            platform = mm.Platform.getPlatformByName('CUDA')
            platform.setPropertyDefaultValue('CUDAPrecision', 'double')
        else:
            raise(ValueError("Invalid platform name"))
    except:
        # Return CPU platform if we can't provide requested platofrm
        print("Warning: Returning '%s' platform instead of requested platform '%s'" % (fallback_platform_name, platform_name))
        return mm.Platform.getPlatformByName(fallback_platform_name)

    return platform
