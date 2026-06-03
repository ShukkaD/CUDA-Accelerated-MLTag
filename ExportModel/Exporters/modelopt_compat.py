import os
import warnings


def patch_modelopt_environment() -> None:
    try:
        nvcc_append_flags = os.environ.get("NVCC_APPEND_FLAGS", "")
        required_nvcc_flag = "-Xcompiler=/Zc:preprocessor"

        if required_nvcc_flag not in nvcc_append_flags:
            os.environ["NVCC_APPEND_FLAGS"] = (
                f"{nvcc_append_flags} {required_nvcc_flag}".strip()
            )
    except Exception:
        warnings.warn("Failed to patch ModelOpt environment variables.")
        pass

    try:
        import torch

        script_module = getattr(torch.jit, "_script", None)
        script_impl = getattr(script_module, "_script_impl", None)
        if script_impl is not None and getattr(torch.jit, "script", None) is not script_impl:
            torch.jit.script = script_impl
    except (ImportError, AttributeError):
        warnings.warn("Failed to patch ModelOpt JIT script behavior.")
        pass

    if os.name != "nt":
        return

    try:
        from setuptools._distutils.compilers.C import msvc as _msvc
        import setuptools._distutils._msvccompiler as _legacy_msvc

        _legacy_msvc._get_vc_env = _msvc._get_vc_env
    except ImportError:
        warnings.warn("Failed to patch ModelOpt MSVC compiler behavior.")
        pass