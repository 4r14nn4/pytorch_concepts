"""Post-hoc analysis of finished runs.

Each script adds this package's parent to ``sys.path`` at import time, so all
three invocations work identically::

    python conceptarium/analysis/run_generative_analysis.py   # from the repo root
    python analysis/run_generative_analysis.py                # from conceptarium/
    python -m analysis.run_generative_analysis                # from conceptarium/

Hydra finds ``conf/`` through each script's ``config_path="../conf"``, which is
resolved relative to the script file rather than the working directory.
"""
