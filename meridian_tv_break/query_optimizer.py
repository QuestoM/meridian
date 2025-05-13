import argparse
import logging
from pathlib import Path

import pandas as pd
from meridian.model.model import load_mmm

from tv_break_optimizer import TVBreakOptimizer, BreakSchedulePlanner  # reuse logic

logger = logging.getLogger(__name__)


def run_query(model_path: Path, programme_grid_path: Path, ad_inventory_path: Path | None, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading frozen posterior from %s", model_path)
    mmm = load_mmm(model_path)
    optimizer = TVBreakOptimizer(mmm)
    planner = BreakSchedulePlanner(mmm, optimizer)

    programme_df = pd.read_csv(programme_grid_path, encoding="utf-8")
    ad_inv_df = pd.read_csv(ad_inventory_path, encoding="utf-8") if ad_inventory_path else None

    logger.info("Generating daily plan for %s", programme_grid_path.name)
    plan = planner.generate_daily_plan(program_schedule=programme_df, ad_inventory=ad_inv_df)

    csv_out = output_dir / f"optimal_break_plan_keshet12_{programme_grid_path.stem}.csv"
    json_out = output_dir / f"summary_{programme_grid_path.stem}.json"

    pd.DataFrame(plan["break_schedule"]).to_csv(csv_out, encoding="utf-8", index=False)
    import json, datetime as dt

    json_out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Outputs written: %s, %s", csv_out, json_out)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")
    parser = argparse.ArgumentParser(description="Load frozen posterior and produce optimal break schedule for Keshet 12")
    parser.add_argument("programme_grid", type=Path)
    parser.add_argument("--ad_inventory", type=Path)
    parser.add_argument("--model_path", type=Path, default=Path("models/tv_break_posterior.pkl"))
    parser.add_argument("--output_dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    run_query(args.model_path, args.programme_grid, args.ad_inventory, args.output_dir)


if __name__ == "__main__":
    main()
