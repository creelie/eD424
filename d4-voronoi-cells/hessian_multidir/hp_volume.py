#!/usr/bin/env python3
"""
The high-precision volume routine used by the scripts in this directory.

There is one copy of it, in core/hp_volume.py.  The scripts here import it
as "from hp_volume import hp_volume", so this file loads that copy under
this name and re-exports what it defines.  Nothing is duplicated.
"""
import importlib.util
import os

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    os.pardir, 'core', 'hp_volume.py')

_spec = importlib.util.spec_from_file_location('_hp_volume_core', _SRC)
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)

find_vertices = _core.find_vertices
tangent_basis_mp = _core.tangent_basis_mp
det4 = _core.det4
hp_volume = _core.hp_volume

__all__ = ['find_vertices', 'tangent_basis_mp', 'det4', 'hp_volume']
