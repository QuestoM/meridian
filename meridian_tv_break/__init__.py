# meridian_tv_break package

"""High-level API for Keshet-12 TV break optimisation.

Modules:
    data_transform  – handles raw XLSX→CSV enrichment.
    train_model     – fits Bayesian MMM and saves frozen posterior.
    query_optimizer – loads posterior and produces optimal break plan.
"""

__version__: str = "0.1.0"
