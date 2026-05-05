"""Compatibility wrapper for the canonical pipeline app module."""

from thesis_pipeline.pipeline.app import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("thesis_pipeline.pipeline.app", run_name="__main__")
