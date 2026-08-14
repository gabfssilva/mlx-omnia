"""The menu bar face of the window: a status item, and a panel that hangs off it.

The daemon is the same one `appv2` talks to — whichever process is resident owns it and the
other attaches, which `mlx_omnia.app.api.daemon` already decides. What is different here is
only the window: 380 pt with no sidebar, four tabs, and a head that never leaves the screen.
"""
