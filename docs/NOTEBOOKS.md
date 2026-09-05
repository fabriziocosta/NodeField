# Notebook guide

The [notebook task chooser](../notebooks/README.md) is the maintained catalog,
including prerequisites and expected outputs. Start with [environment setup](../notebooks/setup.ipynb).

Notebooks are grouped by purpose: synthetic graphs, molecules, campaigns,
experiments, and validation. Experimental workflows are separate from standard
training and sampling entry points. Filenames describe actions within each folder.

Each workflow explains what the next step does before it runs. Shared summaries
and molecule inspection live in the package, while experiment settings remain
visible in the notebook. The two ZINC training paths retain their distinct settings.

Data and configuration paths are resolved from the repository root through
`configure_notebook()`, so opening a notebook in its new subdirectory uses the same
`notebooks/datasets`, `notebooks/configs`, `.artifacts`, and `artifact` locations.
Select the environment prepared by setup and run each notebook's own setup cell.

Saved cell outputs are cleared before commit. Generated datasets, checkpoints,
plots, and campaign state belong in ignored data and artifact folders.
