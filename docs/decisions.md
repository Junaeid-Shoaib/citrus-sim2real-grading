# Decisions

One line per decision: what, why, alternative rejected.

- Plain scripts in src/ instead of an installable package: simpler to run with `uv run python src/x.py`; rejected the uv --package layout as unneeded for a single project.
- uv for environment and packages: lockfile makes versions reproducible; rejected pip + requirements.txt.
