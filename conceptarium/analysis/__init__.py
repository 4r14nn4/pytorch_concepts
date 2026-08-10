"""Post-hoc analysis of finished runs.

Run from the ``conceptarium`` directory, as modules — that puts ``conceptarium``
itself on ``sys.path``, so ``conceptarium.*`` and ``env`` resolve without a path
hack, and Hydra finds ``conf/`` through each script's ``config_path="../conf"``::

    python -m analysis.run_analysis
    python -m analysis.run_generative_analysis
"""
