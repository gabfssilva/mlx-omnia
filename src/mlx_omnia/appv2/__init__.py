"""Quieta — the appv2 window: four places and a gauge, wired to the daemon.

The app opens on the Server; Chat, Models and Benchmark sit beside it. Every screen
draws the daemon's own resources — the event stream fills one Engine, the window
starts a server when nobody answers on the port, and every write goes back over HTTP.
The window knows no engine: `mlx_omnia.app.api` is its whole reach.
"""
