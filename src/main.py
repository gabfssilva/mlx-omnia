"""The bundle's entry point.

`flet build` runs a module named at the root of the packaged app directory, and it is
`src/` that is packaged so `mlx_omnia.app` stays importable as a package. Nothing lives
here but the call: the window is `mlx_omnia.app.main`.
"""

from mlx_omnia.app.main import run

run()
