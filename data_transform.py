#!/usr/bin/env python3
"""CLI utility to standardise/raw-to-enriched transform for TV-break data.

Usage
-----
    python data_transform.py \
        --dayparts Dayparts.xlsx \
        --programmes Programmes.xlsx \
        --spots Spots.xlsx \
        --output_dir data/enriched

If the input files are already enriched UTF-8 CSVs (detected via file
extension), the script simply copies/validates and exits quickly.

The enrichment logic mirrors – but is decoupled from – legacy
`tv_break_data_transformer.py`.  It is deliberately **pure** (no network,
no TensorFlow) to keep training/optimisation split.
"""
# ... existing code ...
