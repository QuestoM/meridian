import pandas as pd
import logging
import argparse
from pathlib import Path
from meridian.model.model import load_mmm
from tv_break_model import TVBreakOptimizer
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------

def run_optimization(
    model_path: Path = Path("models/tv_break_posterior.pkl"),
    programs_path: Path = Path("data/Programmes.csv"),
    spots_path: Optional[Path] = None,
    output_path: Path = Path("optimization_results.csv"),
    *,
    min_viewer_retention: float = 0.7,
    max_breaks_per_hour: int = 3,
    budget: float = 100_000.0,
) -> None:
    """Run TV break allocation optimisation.

    Parameters
    ----------
    model_path
        Pickle containing the fitted MMM posterior.
    programs_path
        CSV with the programme grid to optimise on.
    spots_path
        Optional CSV with the ad spot inventory. If provided, enables daily optimisation and ad assignment.
    output_path
        Destination CSV for the optimised break schedule.
    min_viewer_retention
        Required viewer retention threshold (0‒1).
    max_breaks_per_hour
        Max number of breaks allowed in any hour.
    budget
        Budget available for buying breaks (same units as optimiser expects).
    """

    # Resolve paths eagerly to avoid surprises when launched via cron / Airflow
    model_path = model_path.expanduser().resolve()
    programs_path = programs_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if spots_path:
        spots_path = spots_path.expanduser().resolve()

    # Load the trained model
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    logger.info("Loading model from %s", model_path)
    mmm = load_mmm(model_path)

    # Load program schedule
    logger.info(f"Loading program schedule from {programs_path}")
    programs_df = pd.read_csv(programs_path)

    # Load spots inventory if provided
    ad_inventory_df = None
    if spots_path:
        if not spots_path.exists():
            logger.warning(f"Spot inventory file not found at {spots_path}, proceeding without ad assignment.")
        else:
            logger.info(f"Loading spot inventory from {spots_path}")
            ad_inventory_df = pd.read_csv(spots_path)
            # Basic validation for required ad inventory columns
            required_ad_cols = ['duration', 'remaining_count', 'Campaign'] # Example required cols
            missing_ad_cols = [col for col in required_ad_cols if col not in ad_inventory_df.columns]
            if missing_ad_cols:
                logger.warning(f"Spot inventory missing required columns: {missing_ad_cols}. Ad assignment might fail or be inaccurate.")
                # You might add defaults here if appropriate, e.g., ad_inventory_df['remaining_count'] = ad_inventory_df.get('remaining_count', 1)

    # Clean up and prepare the program schedule
    logger.info("Preparing program schedule for optimization")
    # Parse date and time columns
    if 'Start time' in programs_df.columns:
        programs_df['Start time'] = pd.to_datetime(programs_df['Start time'], errors='coerce')
    if 'End time' in programs_df.columns:
        programs_df['End time'] = pd.to_datetime(programs_df['End time'], errors='coerce')
    if 'Date' in programs_df.columns:
        programs_df['Date'] = pd.to_datetime(programs_df['Date'], format='%d/%m/%Y', errors='coerce')

    # Calculate duration if not present
    if 'Duration' not in programs_df.columns and 'Start time' in programs_df.columns and 'End time' in programs_df.columns:
        programs_df['Duration'] = (programs_df['End time'] - programs_df['Start time']).dt.total_seconds() / 60  # Duration in minutes

    # Add program_type if not present
    if 'program_type' not in programs_df.columns and 'Title' in programs_df.columns:
        # Simple categorization function
        def categorize(title):
            title = str(title).lower()
            if any(kw in title for kw in ['חדשות', 'כותרות', 'מהדורה']):
                return 'News'
            elif any(kw in title for kw in ['סדרת', 'סדרה', 'דרמה']):
                return 'Drama'
            elif any(kw in title for kw in ['קומדיה', 'בידור']):
                return 'Comedy'
            elif any(kw in title for kw in ['ספורט', 'כדורגל', 'כדורסל']):
                return 'Sports'
            elif any(kw in title for kw in ['תעודה', 'תיעודי']):
                return 'Documentary'
            elif any(kw in title for kw in ['ריאליטי']):
                return 'Reality'
            elif any(kw in title for kw in ['פרומו', 'פרסומות']):
                return 'Promo'
            else:
                return 'Other'

        programs_df['program_type'] = programs_df['Title'].apply(categorize)

    # Add prime_time column if not present
    if 'prime_time' not in programs_df.columns and 'Start time' in programs_df.columns:
        programs_df['prime_time'] = ((programs_df['Start time'].dt.hour >= 18) &
                                    (programs_df['Start time'].dt.hour < 23)).astype(float)

    # Add viewing_points if not present (example values based on time)
    if 'viewing_points' not in programs_df.columns and 'Start time' in programs_df.columns:
        def calculate_viewing_points(row):
            hour = row['Start time'].hour if pd.notna(row['Start time']) else 12
            prime_time = 18 <= hour < 23
            return 3.0 if prime_time else (1.5 if hour >= 12 else 1.0)

        programs_df['viewing_points'] = programs_df.apply(calculate_viewing_points, axis=1)

    # Ensure we have the required columns
    required_cols = ['program_type', 'Duration', 'prime_time', 'viewing_points']
    missing_cols = [col for col in required_cols if col not in programs_df.columns]
    if missing_cols:
        logger.warning(f"Missing required columns in program schedule: {missing_cols}")
        for col in missing_cols:
            if col == 'Duration':
                programs_df['Duration'] = 60.0  # Default duration in minutes
            elif col == 'prime_time':
                programs_df['prime_time'] = 0.0  # Default not prime time
            elif col == 'viewing_points':
                programs_df['viewing_points'] = 1.0  # Default viewing points
            elif col == 'program_type':
                programs_df['program_type'] = 'Other'  # Default program type

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    # Initialize the optimizer
    logger.info("Initializing TVBreakOptimizer")
    optimizer = TVBreakOptimizer(mmm)

    # Run optimization - Use daily if spots provided, otherwise fallback
    logger.info("Running optimization")
    final_schedule_df = pd.DataFrame()
    try:
        if ad_inventory_df is not None:
            logger.info("Running daily schedule optimization with ad assignment.")
            # Ensure necessary columns for daily schedule are present or defaulted in programs_df
            # Example: Ensure 'day' exists if optimize_daily_schedule expects it
            if 'day_name' not in programs_df.columns and 'Date' in programs_df.columns:
                programs_df['day_name'] = programs_df['Date'].dt.day_name()
            elif 'day_name' not in programs_df.columns:
                logger.warning("Missing 'day_name' or 'Date' in program schedule, cannot run daily optimization robustly. Assuming single day.")
                # Handle default day logic if needed, or let optimizer handle it

            # Call optimize_daily_schedule which includes ad assignment
            final_schedule_df = optimizer.optimize_daily_schedule(
                program_schedule=programs_df,
                ad_inventory=ad_inventory_df,
                min_viewer_retention=min_viewer_retention,
                max_breaks_per_hour=max_breaks_per_hour,
                budget=budget
            )
        else:
            logger.info("Running basic allocation optimization (no ad inventory provided).")
            optimization_results = optimizer.optimize_allocation(
                program_schedule=programs_df,
                min_viewer_retention=min_viewer_retention,
                max_breaks_per_hour=max_breaks_per_hour,
                budget=budget
            )
            if 'break_schedule' in optimization_results and isinstance(optimization_results['break_schedule'], pd.DataFrame):
                final_schedule_df = optimization_results['break_schedule']
            else:
                logger.warning("Basic optimization did not return a valid break schedule")

        if not final_schedule_df.empty:
            # --- Output Formatting ---
            logger.info("Formatting output schedule...")

            # Example formatting to resemble Spots - sample1day.csv
            # This part needs careful mapping based on actual output columns vs target columns

            # 1. Explode ad_assignments if they exist
            if 'ad_assignments' in final_schedule_df.columns:
                # Ensure it's list type, handle NaN/None
                final_schedule_df['ad_assignments'] = final_schedule_df['ad_assignments'].apply(lambda x: x if isinstance(x, list) else [])
                exploded_df = final_schedule_df.explode('ad_assignments', ignore_index=True)
                # Normalize the ad_assignments dict into columns
                ad_details_df = pd.json_normalize(exploded_df['ad_assignments']).fillna({'ad_id': pd.NA, 'duration': pd.NA, 'name': pd.NA})
                ad_details_df = ad_details_df.add_prefix('ad_') # Prefix ad columns
                formatted_df = pd.concat([exploded_df.drop(columns=['ad_assignments']), ad_details_df], axis=1)
            else:
                # If no ad assignments, just use the break schedule
                formatted_df = final_schedule_df.copy()
                # Add placeholder ad columns if needed for consistency
                formatted_df['ad_id'] = pd.NA
                formatted_df['ad_duration'] = pd.NA
                formatted_df['ad_name'] = pd.NA

            # 2. Map/Rename/Calculate columns to match target (example mappings)
            # Target columns from Spots - sample1day.csv:
            # Unnamed: 0, Campaign, Channel, Date, Start time, Duration, Spot type, Promotion,
            # Pos. Block 1, Spots Block 1, Title, TVR, Date_dt, Start_dt, End_dt,
            # hour_of_day, quarter_id, advertiser_id, break_id, position_in_break,
            # channel, base_rate, prime_premium, pos1, pos2, pos3, pos_last,
            # adv_premium, position_premium, total_premium, revenue_ils,
            # is_target_channel, include_as_media, competitor_flag

            # --- Potential Renames/Mappings (adjust based on actual output col names) ---
            rename_map = {
                'ad_name': 'Campaign',       # Assuming ad_name is the campaign name
                'ad_duration': 'Duration',   # Assuming ad_duration is the spot duration
                'ad_id': 'advertiser_id',    # Assuming ad_id maps to advertiser_id (or campaign?)
                'predicted_revenue': 'revenue_ils', # Assuming revenue is in ILS
                'program_type': 'Title',     # Maybe map program type or keep original Title if available
                # Add more mappings...
            }
            formatted_df = formatted_df.rename(columns=rename_map)

            # --- Add Missing Placeholder Columns ---
            target_cols = [ # List the columns from Spots - sample1day.csv
                'Campaign', 'Channel', 'Date', 'Start time', 'Duration', 'Spot type', 'Promotion',
                'Pos. Block 1', 'Spots Block 1', 'Title', 'TVR', 'Date_dt', 'Start_dt', 'End_dt',
                'hour_of_day', 'quarter_id', 'advertiser_id', 'break_id', 'position_in_break',
                'channel', 'base_rate', 'prime_premium', 'pos1', 'pos2', 'pos3', 'pos_last',
                'adv_premium', 'position_premium', 'total_premium', 'revenue_ils',
                'is_target_channel', 'include_as_media', 'competitor_flag'
            ]
            for col in target_cols:
                if col not in formatted_df.columns:
                    formatted_df[col] = pd.NA # Or calculate if possible, e.g., 'Date' from Start_dt

            # --- Calculate/Derive Columns ---
            # Example: Derive 'Start time' (time part) from a datetime column if available
            # if 'Start_dt' in formatted_df.columns: # Assuming a datetime column exists
            #    formatted_df['Start time'] = formatted_df['Start_dt'].dt.strftime('%H:%M:%S')
            #    formatted_df['Date'] = formatted_df['Start_dt'].dt.strftime('%d/%m/%Y') # Example Date format

            # --- Reorder Columns ---
            # Ensure only columns present in the DataFrame are selected
            final_cols_ordered = [col for col in target_cols if col in formatted_df.columns]
            formatted_df = formatted_df[final_cols_ordered]

            # --- Save the formatted results ---
            formatted_df.to_csv(output_path, index=False, encoding='utf-8-sig') # Use utf-8-sig for Excel compatibility
            logger.info(f"Formatted break schedule saved to %s", output_path)

            # --- Log Summary ---
            total_spots = len(formatted_df) if 'ad_id' in formatted_df.columns else 0
            total_revenue_final = formatted_df['revenue_ils'].sum() if 'revenue_ils' in formatted_df.columns else 0
            logger.info("Optimization successful!")
            logger.info(f"Total spots scheduled: {total_spots}")
            logger.info(f"Total predicted revenue (ILS): {total_revenue_final:,.2f}")

        else:
            logger.warning("Optimization produced an empty schedule.")

    except Exception:
        logger.error("Optimization failed", exc_info=True)
        raise

# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize TV break allocation")
    parser.add_argument("--model-path", type=Path, default="models/tv_break_posterior.pkl", help="Pickle with trained MMM posterior")
    parser.add_argument("--programs-path", type=Path, default="data/Programmes.csv", help="CSV programme grid")
    parser.add_argument("--spots-inventory", type=Path, default=None, help="Optional CSV ad spot inventory")
    parser.add_argument("--output-path", type=Path, default="optimization_results.csv", help="Destination CSV for break schedule")
    parser.add_argument("--min-retention", type=float, default=0.7, help="Minimum viewer retention (0-1)")
    parser.add_argument("--max-breaks-per-hour", type=int, default=3, help="Maximum breaks per hour")
    parser.add_argument("--budget", type=float, default=100_000.0, help="Budget for optimisation")

    args = parser.parse_args()

    run_optimization(
        model_path=args.model_path,
        programs_path=args.programs_path,
        spots_path=args.spots_inventory,
        output_path=args.output_path,
        min_viewer_retention=args.min_retention,
        max_breaks_per_hour=args.max_breaks_per_hour,
        budget=args.budget,
    )
