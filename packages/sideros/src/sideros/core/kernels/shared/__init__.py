"""Format machinery shared between kernel facades: Metal source, decode helpers,
layouts and headers.

Nothing here builds, dispatches or decides — a strategy imports a decode header or a
re-layout from this package and keeps its own predicate, epilogue and dispatch. One
module (or package) per quantization format.
"""
