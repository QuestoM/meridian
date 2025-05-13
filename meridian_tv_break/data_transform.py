import argparse
import logging
from pathlib import Path
from typing import Tuple
import pandas as pd

# Re-use legacy transformer to avoid code duplication
from tv_break_data_transformer import TVBreakDataTransformer

logger = logging.getLogger(__name__)

ENCODING = "utf-8"


def _is_enriched_dayparts(df: pd.DataFrame) -> bool:
    return "datetime" in df.columns and "hour_of_day" in df.columns

def _is_enriched_programmes(df: pd.DataFrame) -> bool:
    return "program_type" in df.columns and "Start_datetime" in df.columns if df.columns.str.contains("Start_datetime").any() else "programme_id" in df.columns

def _is_enriched_spots(df: pd.DataFrame) -> bool:
    return "avg_tvr" in df.columns or "spot_id" in df.columns


def detect_enriched(dayparts_path: Path, programmes_path: Path, spots_path: Path) -> Tuple[bool, bool, bool]:
    """Return three booleans whether each CSV is already enriched."""
    try:
        dp = pd.read_csv(dayparts_path, nrows=5, encoding=ENCODING)
        pr = pd.read_csv(programmes_path, nrows=5, encoding=ENCODING)
        sp = pd.read_csv(spots_path, nrows=5, encoding=ENCODING)
    except UnicodeDecodeError:
        # Fallback without explicit encoding
        dp = pd.read_csv(dayparts_path, nrows=5)
        pr = pd.read_csv(programmes_path, nrows=5)
        sp = pd.read_csv(spots_path, nrows=5)
    return (_is_enriched_dayparts(dp), _is_enriched_programmes(pr), _is_enriched_spots(sp))


def enrich_and_save(dayparts_path: Path, programmes_path: Path, spots_path: Path, output_dir: Path):
    """Run legacy transformer on raw XLSX or CSV and persist enriched CSV into *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # tv_break_data_transformer expects Excel; patch read_excel for CSV
    original_read_excel = pd.read_excel

    def adaptive_read_excel(path, *args, **kwargs):
        path = Path(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, *args, **kwargs)
        return original_read_excel(path, *args, **kwargs)

    pd.read_excel = adaptive_read_excel  # type: ignore

    try:
        transformer = TVBreakDataTransformer(dayparts_path, programmes_path, spots_path)
        result = transformer.transform_data()
    finally:
        pd.read_excel = original_read_excel  # restore

    if result is None or result.get("raw_data") is None:
        raise RuntimeError("Data enrichment failed – transformer returned None")

    raw = result["raw_data"]
    # Write enriched CSVs
    raw["dayparts"].to_csv(output_dir / "Dayparts.csv", encoding=ENCODING, index=False)
    raw["programmes"].to_csv(output_dir / "Programmes.csv", encoding=ENCODING, index=False)
    raw["spots"].to_csv(output_dir / "Spots.csv", encoding=ENCODING, index=False)

    logger.info("Enriched CSVs written to %s", output_dir)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")
    parser = argparse.ArgumentParser(description="Detect/enrich TV break raw data and output enriched CSVs (UTF-8).")
    parser.add_argument("--dayparts", type=Path, default=Path("data/Dayparts.csv"))
    parser.add_argument("--programmes", type=Path, default=Path("data/Programmes.csv"))
    parser.add_argument("--spots", type=Path, default=Path("data/Spots.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/enriched"))
    parser.add_argument("--skip_enrich", action="store_true", help="Force skip enrichment even if files look raw")

    args = parser.parse_args()

    logging.debug("Args: %s", args)
    enriched_flags = detect_enriched(args.dayparts, args.programmes, args.spots)
    logger.info("Detected enriched flags: dayparts=%s, programmes=%s, spots=%s", *enriched_flags)

    if all(enriched_flags) or args.skip_enrich:
        logger.info("All inputs appear to be already enriched – copying to output directory")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for src in [args.dayparts, args.programmes, args.spots]:
            dest = args.output_dir / src.name.replace(".csv", ".csv")
            if src != dest:
                dest.write_bytes(src.read_bytes())
        return

    logger.info("Detected raw inputs – running enrichment pipeline")
    enrich_and_save(args.dayparts, args.programmes, args.spots, args.output_dir)


if __name__ == "__main__":
    main()
