#!/usr/bin/env python

import os
import sys
import build_helpers

# If called recursively in superbuild, a global persistent LegacyUninstaller will be returned.
lu = build_helpers.get_global_legacy_uninstaller()

incdir = os.path.join(os.environ['HOME'], 'include')
lu.add_glob(incdir, 'simd_helpers.hpp')
lu.add_glob(incdir, 'simd_helpers/*.hpp')
lu.add_glob(incdir, 'simd_helpers')

# If called recursively in superbuild, run() will not be called here.
if __name__ == '__main__':
    lu.run()
