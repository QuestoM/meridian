"""Transform TV schedule, ratings, and spot logs into Meridian input data.

The original project started as a quick adaptation of Google's Meridian package.
This module keeps that adapter explicit: TV commercial break seconds are treated
as the paid-media signal, while viewer retention is the KPI.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
from meridian.data import input_data

logger = logging.getLogger(__name__)

PROGRAM_TYPES = (
    "News",
    "Drama",
    "Comedy",
    "Reality",
    "Documentary",
    "Sports",
    "Promo",
    "Other",
)
POSITIONS = ("Early", "Middle", "Late")
BREAK_TYPES = ("Short", "Medium", "Long")

METADATA_COLUMNS = {
    "",
    "Unnamed: 0",
    "date",
    "Date",
    "Dates",
    "time",
    "Timebands",
    "datetime",
    "hour_of_day",
    "quarter_id",
    "is_prime_time",
    "day_name",
    "is_weekend",
    "season_id",
    "col_0",
}


class TVBreakDataTransformer:
    """Build Meridian-compatible data from TV broadcast source files."""

    def __init__(
        self,
        dayparts_path: str | Path,
        programmes_path: str | Path,
        spots_path: str | Path,
        time_freq: str = "D",
    ) -> None:
        self.dayparts_path = Path(dayparts_path)
        self.programmes_path = Path(programmes_path)
        self.spots_path = Path(spots_path)
        self.time_freq = time_freq

    def transform_data(self) -> dict[str, object]:
        """Load, normalize, aggregate, and convert source data."""
        dayparts_df = self._read_table(self.dayparts_path)
        programmes_df = self._read_table(self.programmes_path)
        spots_df = self._read_table(self.spots_path)

        dayparts_clean = self._preprocess_dayparts(dayparts_df)
        programmes_clean = self._preprocess_programmes(programmes_df)
        spots_clean = self._preprocess_spots(spots_df)

        breaks_df = self._identify_commercial_breaks(spots_clean)
        breaks_with_programs = self._match_breaks_with_programs(
            breaks_df, programmes_clean
        )
        breaks_with_metrics = self._calculate_viewership_metrics(
            breaks_with_programs, dayparts_clean
        )
        breaks_with_revenue = self._calculate_revenue_metrics(breaks_with_metrics)

        aggregated_data = self._create_aggregated_data(
            breaks_with_revenue, programmes_clean, dayparts_clean
        )
        meridian_data = self._create_meridian_data(aggregated_data)

        return {
            "raw_data": {
                "dayparts": dayparts_clean,
                "programmes": programmes_clean,
                "spots": spots_clean,
                "breaks": breaks_with_revenue,
            },
            "aggregated_data": aggregated_data,
            "meridian_data": meridian_data,
        }

    @staticmethod
    def _read_table(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if suffix == ".csv":
            return pd.read_csv(path, encoding="utf-8-sig")
        raise ValueError(f"Unsupported input format for {path}")

    @staticmethod
    def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
        available = set(columns)
        return next((candidate for candidate in candidates if candidate in available), None)

    @staticmethod
    def _parse_datetime_pair(df: pd.DataFrame, date_col: str, time_col: str) -> pd.Series:
        combined = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
        return pd.to_datetime(combined, dayfirst=True, errors="coerce")

    @staticmethod
    def _break_type_from_seconds(seconds: float) -> str:
        if seconds < 60:
            return "Short"
        if seconds < 120:
            return "Medium"
        return "Long"

    @staticmethod
    def _position_from_ratio(ratio: float) -> str:
        if ratio <= 0.33:
            return "Early"
        if ratio <= 0.66:
            return "Middle"
        return "Late"

    def _preprocess_dayparts(self, dayparts_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize ratings rows to date, time, datetime, and channel TVR columns."""
        df = dayparts_df.copy()
        df.columns = [str(col).strip() if not pd.isna(col) else f"col_{idx}" for idx, col in enumerate(df.columns)]

        clean_df = pd.DataFrame()
        date_col = self._first_existing(df.columns, ("Dates", "Date", "date", "Date_dt"))
        time_col = self._first_existing(df.columns, ("Timebands", "time", "Start time"))

        if date_col is None:
            raise ValueError("Dayparts file is missing a date column")

        clean_df["date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce").dt.date

        if "datetime" in df.columns:
            clean_df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True, errors="coerce")
        elif time_col is not None:
            clean_df["time"] = df[time_col].astype(str).str.strip()
            clean_df["datetime"] = self._parse_datetime_pair(clean_df, "date", "time")
        else:
            raise ValueError("Dayparts file is missing a time or datetime column")

        if "time" not in clean_df.columns:
            clean_df["time"] = clean_df["datetime"].dt.strftime("%H:%M")

        channel_cols = [
            col
            for col in df.columns
            if col not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
        ]
        if not channel_cols:
            channel_cols = [col for col in df.columns if "TVR" in col]

        for channel in channel_cols:
            clean_df[channel] = pd.to_numeric(df[channel], errors="coerce")

        clean_df = clean_df.dropna(subset=["date", "datetime"]).reset_index(drop=True)
        if clean_df.empty:
            raise ValueError("Dayparts preprocessing produced no valid rows")
        return clean_df

    def _preprocess_programmes(self, programmes_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize programme grid columns used by matching and optimization."""
        df = programmes_df.copy()
        df.columns = [str(col).strip() for col in df.columns]

        if "Start_datetime" in df.columns:
            df["Start time"] = pd.to_datetime(df["Start_datetime"], dayfirst=True, errors="coerce")
        elif "Start_dt" in df.columns:
            df["Start time"] = pd.to_datetime(df["Start_dt"], dayfirst=True, errors="coerce")
        elif {"Date", "Start time"}.issubset(df.columns):
            df["Start time"] = self._parse_datetime_pair(df, "Date", "Start time")
        elif "Start time" in df.columns:
            df["Start time"] = pd.to_datetime(df["Start time"], dayfirst=True, errors="coerce")

        if "End_datetime" in df.columns:
            df["End time"] = pd.to_datetime(df["End_datetime"], dayfirst=True, errors="coerce")
        elif "End_dt" in df.columns:
            df["End time"] = pd.to_datetime(df["End_dt"], dayfirst=True, errors="coerce")
        elif {"Date", "End time"}.issubset(df.columns):
            df["End time"] = self._parse_datetime_pair(df, "Date", "End time")
        elif "End time" in df.columns:
            df["End time"] = pd.to_datetime(df["End time"], dayfirst=True, errors="coerce")

        if {"Start time", "End time"}.issubset(df.columns):
            crosses_midnight = df["End time"] < df["Start time"]
            df.loc[crosses_midnight, "End time"] = df.loc[crosses_midnight, "End time"] + pd.Timedelta(days=1)

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date
        elif "Start time" in df.columns:
            df["Date"] = df["Start time"].dt.date

        if "Duration" not in df.columns and {"Start time", "End time"}.issubset(df.columns):
            df["Duration"] = (df["End time"] - df["Start time"]).dt.total_seconds()
        else:
            df["Duration"] = pd.to_numeric(df.get("Duration", 0), errors="coerce")

        if "program_type" not in df.columns:
            if "programme_type" in df.columns:
                df["program_type"] = df["programme_type"].fillna("Other")
            else:
                df["program_type"] = self._categorize_programs(df)
        df["program_type"] = df["program_type"].fillna("Other").astype(str)

        if "prime_time" not in df.columns:
            if "is_prime_time" in df.columns:
                df["prime_time"] = df["is_prime_time"].astype(str).str.lower().isin({"true", "1", "yes"})
            elif "Start time" in df.columns:
                df["prime_time"] = df["Start time"].dt.hour.between(18, 22, inclusive="both")
            else:
                df["prime_time"] = False

        if "viewing_points" not in df.columns:
            if "TVR" in df.columns:
                df["viewing_points"] = pd.to_numeric(df["TVR"], errors="coerce").fillna(1.0)
            elif "Start time" in df.columns:
                df["viewing_points"] = df["Start time"].dt.hour.apply(
                    lambda hour: 3.0 if 18 <= hour < 23 else (1.5 if hour >= 12 else 1.0)
                )
            else:
                df["viewing_points"] = 1.0

        return df.dropna(subset=["Start time", "End time"]).reset_index(drop=True)

    def _preprocess_spots(self, spots_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize spot logs and derive a concrete start/end datetime."""
        df = spots_df.copy()
        df.columns = [str(col).strip() for col in df.columns]

        if "Start_dt" in df.columns:
            df["Start time"] = pd.to_datetime(df["Start_dt"], dayfirst=True, errors="coerce")
        elif "Start_datetime" in df.columns:
            df["Start time"] = pd.to_datetime(df["Start_datetime"], dayfirst=True, errors="coerce")
        elif {"Date", "Start time"}.issubset(df.columns):
            df["Start time"] = self._parse_datetime_pair(df, "Date", "Start time")
        elif "Start time" in df.columns:
            df["Start time"] = pd.to_datetime(df["Start time"], dayfirst=True, errors="coerce")

        df["Duration"] = pd.to_numeric(df.get("Duration", 0), errors="coerce").fillna(0)
        df["End time"] = df["Start time"] + pd.to_timedelta(df["Duration"], unit="s")

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date
        else:
            df["Date"] = df["Start time"].dt.date

        if "TVR" in df.columns:
            df["TVR"] = pd.to_numeric(df["TVR"], errors="coerce")
        else:
            df["TVR"] = np.nan

        required = ["Channel", "Start time", "End time", "Duration"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Spots file missing required columns: {missing}")

        return df.dropna(subset=["Channel", "Start time", "End time"]).reset_index(drop=True)

    def _categorize_programs(self, programmes_df: pd.DataFrame) -> pd.Series:
        """Categorize programmes from title text when no type column exists."""
        title_col = self._first_existing(programmes_df.columns, ("Title", "Program", "Programme", "Name"))
        if title_col is None:
            return pd.Series(["Other"] * len(programmes_df), index=programmes_df.index)

        def categorize(title: object) -> str:
            value = str(title).lower()
            if any(keyword in value for keyword in ("חדשות", "כותרות", "מהדורה", "news")):
                return "News"
            if any(keyword in value for keyword in ("סדרה", "דרמה", "series", "drama")):
                return "Drama"
            if any(keyword in value for keyword in ("קומד", "בידור", "comedy")):
                return "Comedy"
            if any(keyword in value for keyword in ("ספורט", "כדורגל", "כדורסל", "sports")):
                return "Sports"
            if any(keyword in value for keyword in ("תעודה", "תיעודי", "documentary")):
                return "Documentary"
            if any(keyword in value for keyword in ("ריאליטי", "reality")):
                return "Reality"
            if any(keyword in value for keyword in ("פרומו", "promo")):
                return "Promo"
            return "Other"

        return programmes_df[title_col].apply(categorize)

    def _identify_commercial_breaks(self, spots_df: pd.DataFrame) -> pd.DataFrame:
        """Group adjacent spots into commercial breaks."""
        if spots_df.empty:
            return pd.DataFrame()

        spots = spots_df.sort_values(["Channel", "Start time"]).reset_index(drop=False)
        breaks: list[dict[str, object]] = []

        for channel, channel_spots in spots.groupby("Channel", sort=False):
            current_break: list[pd.Series] = []

            for _, spot in channel_spots.iterrows():
                if not current_break:
                    current_break = [spot]
                    continue

                last_spot = current_break[-1]
                gap_seconds = (spot["Start time"] - last_spot["End time"]).total_seconds()
                if gap_seconds <= 15:
                    current_break.append(spot)
                else:
                    self._append_break(breaks, channel, current_break)
                    current_break = [spot]

            self._append_break(breaks, channel, current_break)

        breaks_df = pd.DataFrame(breaks)
        if not breaks_df.empty:
            breaks_df["break_type"] = breaks_df["break_duration"].apply(self._break_type_from_seconds)
        return breaks_df

    @staticmethod
    def _append_break(breaks: list[dict[str, object]], channel: str, spots: list[pd.Series]) -> None:
        if len(spots) < 2:
            return

        break_start = spots[0]["Start time"]
        break_end = spots[-1]["End time"]
        break_duration = (break_end - break_start).total_seconds()
        breaks.append(
            {
                "channel": channel,
                "break_start": break_start,
                "break_end": break_end,
                "break_duration": break_duration,
                "num_spots": len(spots),
                "spots": [spot.get("Campaign", spot.get("Title", "")) for spot in spots],
                "spot_ids": [int(spot.get("index", idx)) for idx, spot in enumerate(spots)],
                "avg_tvr": pd.Series([spot.get("TVR", np.nan) for spot in spots]).mean(),
            }
        )

    def _match_breaks_with_programs(self, breaks_df: pd.DataFrame, programmes_df: pd.DataFrame) -> pd.DataFrame:
        """Attach containing programme metadata to each commercial break."""
        if breaks_df.empty:
            return breaks_df

        result = breaks_df.copy()
        defaults = {
            "program_title": None,
            "program_type": "Other",
            "program_duration": np.nan,
            "seconds_into_program": np.nan,
            "relative_position": np.nan,
            "position_category": "Middle",
            "program_tvr": np.nan,
            "prime_time": False,
            "viewing_points": 1.0,
        }
        for column, default in defaults.items():
            result[column] = default

        for idx, break_info in result.iterrows():
            matching = programmes_df[
                (programmes_df["Channel"] == break_info["channel"])
                & (programmes_df["Start time"] <= break_info["break_start"])
                & (programmes_df["End time"] >= break_info["break_end"])
            ]
            if matching.empty:
                continue

            program = matching.iloc[0]
            duration = float(program.get("Duration", np.nan))
            seconds_into_program = (break_info["break_start"] - program["Start time"]).total_seconds()
            ratio = seconds_into_program / duration if duration and duration > 0 else np.nan

            result.at[idx, "program_title"] = program.get("Title")
            result.at[idx, "program_type"] = program.get("program_type", "Other")
            result.at[idx, "program_duration"] = duration
            result.at[idx, "seconds_into_program"] = seconds_into_program
            result.at[idx, "relative_position"] = ratio
            result.at[idx, "position_category"] = self._position_from_ratio(ratio) if pd.notna(ratio) else "Middle"
            result.at[idx, "program_tvr"] = program.get("TVR", np.nan)
            result.at[idx, "prime_time"] = bool(program.get("prime_time", False))
            result.at[idx, "viewing_points"] = float(program.get("viewing_points", 1.0) or 1.0)

        return result

    def _calculate_viewership_metrics(self, breaks_df: pd.DataFrame, dayparts_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pre, during, and post break retention signals."""
        if breaks_df.empty:
            return breaks_df

        result = breaks_df.copy()
        result["pre_break_tvr"] = np.nan
        result["during_break_tvr"] = pd.to_numeric(result["avg_tvr"], errors="coerce")
        result["post_break_tvr"] = np.nan
        result["viewer_retention"] = np.nan
        result["viewer_recovery"] = np.nan

        for idx, break_info in result.iterrows():
            channel_col = break_info["channel"]
            if channel_col not in dayparts_df.columns:
                continue

            pre_window = dayparts_df[
                (dayparts_df["datetime"] >= break_info["break_start"] - pd.Timedelta(minutes=5))
                & (dayparts_df["datetime"] < break_info["break_start"])
            ]
            post_window = dayparts_df[
                (dayparts_df["datetime"] > break_info["break_end"])
                & (dayparts_df["datetime"] <= break_info["break_end"] + pd.Timedelta(minutes=5))
            ]
            during_window = dayparts_df[
                (dayparts_df["datetime"] >= break_info["break_start"])
                & (dayparts_df["datetime"] <= break_info["break_end"])
            ]

            if not pre_window.empty:
                result.at[idx, "pre_break_tvr"] = pre_window[channel_col].mean()
            if not during_window.empty and pd.isna(result.at[idx, "during_break_tvr"]):
                result.at[idx, "during_break_tvr"] = during_window[channel_col].mean()
            if not post_window.empty:
                result.at[idx, "post_break_tvr"] = post_window[channel_col].mean()

            pre = result.at[idx, "pre_break_tvr"]
            during = result.at[idx, "during_break_tvr"]
            post = result.at[idx, "post_break_tvr"]

            if pd.notna(pre) and pd.notna(during) and pre > 0:
                result.at[idx, "viewer_retention"] = min(max(during / pre, 0.0), 1.5)
            if pd.notna(during) and pd.notna(post) and during > 0:
                result.at[idx, "viewer_recovery"] = min(max(post / during, 0.0), 2.0)

        return result

    def _calculate_revenue_metrics(self, breaks_df: pd.DataFrame) -> pd.DataFrame:
        """Estimate revenue at the break level using available rate-card columns."""
        if breaks_df.empty:
            return breaks_df

        result = breaks_df.copy()
        base_rate = pd.to_numeric(result.get("base_rate", 1000), errors="coerce").fillna(1000)
        if "base_rate" not in result.columns:
            base_rate = pd.Series(1000.0, index=result.index)

        prime_multiplier = np.where(result["prime_time"].astype(bool), 1.5, 1.0)
        tvr = pd.to_numeric(result["during_break_tvr"], errors="coerce").fillna(
            pd.to_numeric(result["viewing_points"], errors="coerce").fillna(1.0)
        )
        minutes = pd.to_numeric(result["break_duration"], errors="coerce").fillna(0) / 60.0

        result["estimated_revenue"] = base_rate * tvr * prime_multiplier * minutes
        result["revenue_per_minute"] = np.divide(
            result["estimated_revenue"],
            minutes.replace(0, np.nan),
        ).fillna(0)
        return result

    def _create_aggregated_data(
        self,
        breaks_df: pd.DataFrame,
        programmes_df: pd.DataFrame,
        dayparts_df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """Aggregate source data to daily Meridian dimensions."""
        dates = self._collect_dates(breaks_df, programmes_df, dayparts_df)
        date_range = pd.date_range(start=min(dates), end=max(dates), freq=self.time_freq)
        time_periods = pd.DataFrame({"date": date_range.date})
        time_periods["day_of_week"] = time_periods["date"].apply(lambda day: pd.Timestamp(day).day_name())
        time_periods["is_weekend"] = time_periods["day_of_week"].isin(["Saturday", "Sunday"])

        breaks_data = self._aggregate_breaks(breaks_df)
        programs_data = self._aggregate_programmes(programmes_df)
        viewership_data = self._aggregate_viewership(dayparts_df)

        return {
            "time_periods": time_periods,
            "breaks": breaks_data,
            "programs": programs_data,
            "viewership": viewership_data,
        }

    @staticmethod
    def _collect_dates(
        breaks_df: pd.DataFrame,
        programmes_df: pd.DataFrame,
        dayparts_df: pd.DataFrame,
    ) -> list[pd.Timestamp]:
        dates: list[pd.Timestamp] = []
        if not breaks_df.empty:
            dates.extend(pd.to_datetime(breaks_df["break_start"]).dt.normalize().tolist())
        if "Date" in programmes_df.columns:
            dates.extend(pd.to_datetime(programmes_df["Date"], errors="coerce").dropna().dt.normalize().tolist())
        if "date" in dayparts_df.columns:
            dates.extend(pd.to_datetime(dayparts_df["date"], errors="coerce").dropna().dt.normalize().tolist())

        cleaned = [date for date in dates if pd.notna(date) and 2010 <= date.year <= 2100]
        if not cleaned:
            start = pd.Timestamp("2024-01-01")
            return [start, start + pd.Timedelta(days=1)]
        if len(set(cleaned)) == 1:
            only = cleaned[0]
            return [only, only + pd.Timedelta(days=1)]
        return sorted(set(cleaned))

    @staticmethod
    def _aggregate_breaks(breaks_df: pd.DataFrame) -> pd.DataFrame:
        if breaks_df.empty:
            return pd.DataFrame()

        df = breaks_df.copy()
        df["date"] = pd.to_datetime(df["break_start"]).dt.date
        df["program_type"] = df["program_type"].fillna("Other")
        df["position_category"] = df["position_category"].fillna("Middle")
        df["break_type"] = df["break_type"].fillna("Medium")

        grouped = df.groupby(
            ["date", "channel", "program_type", "position_category", "break_type"],
            dropna=False,
        )
        return grouped.agg(
            num_breaks=("break_duration", "size"),
            total_break_duration=("break_duration", "sum"),
            avg_break_duration=("break_duration", "mean"),
            total_spots=("num_spots", "sum"),
            avg_viewer_retention=("viewer_retention", "mean"),
            avg_viewer_recovery=("viewer_recovery", "mean"),
            total_revenue=("estimated_revenue", "sum"),
            avg_revenue_per_minute=("revenue_per_minute", "mean"),
            avg_viewing_points=("viewing_points", "mean"),
        ).reset_index()

    @staticmethod
    def _aggregate_programmes(programmes_df: pd.DataFrame) -> pd.DataFrame:
        if programmes_df.empty:
            return pd.DataFrame()

        df = programmes_df.copy()
        df["date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        grouped = df.groupby(["date", "Channel", "program_type"], dropna=False)
        return grouped.agg(
            num_programs=("program_type", "size"),
            total_program_duration=("Duration", "sum"),
            avg_program_duration=("Duration", "mean"),
            avg_program_tvr=("viewing_points", "mean"),
            num_prime_time_programs=("prime_time", "sum"),
        ).reset_index().rename(columns={"Channel": "channel"})

    @staticmethod
    def _aggregate_viewership(dayparts_df: pd.DataFrame) -> pd.DataFrame:
        if dayparts_df.empty:
            return pd.DataFrame()

        df = dayparts_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        channel_cols = [col for col in df.columns if col not in {"date", "time", "datetime"}]
        rows: list[dict[str, object]] = []

        for (date, group) in df.groupby("date", dropna=True):
            prime = group[group["datetime"].dt.hour.between(18, 22, inclusive="both")]
            for channel in channel_cols:
                rows.append(
                    {
                        "date": date,
                        "channel": channel,
                        "avg_viewership": pd.to_numeric(group[channel], errors="coerce").mean(),
                        "peak_viewership": pd.to_numeric(group[channel], errors="coerce").max(),
                        "prime_time_viewership": pd.to_numeric(prime[channel], errors="coerce").mean()
                        if not prime.empty
                        else np.nan,
                    }
                )
        return pd.DataFrame(rows)

    def _create_meridian_data(self, aggregated_data: dict[str, pd.DataFrame]):
        """Convert aggregated TV data into Meridian's InputData object."""
        time_periods = aggregated_data["time_periods"]
        breaks_data = aggregated_data["breaks"]
        programs_data = aggregated_data["programs"]

        if time_periods.empty or breaks_data.empty:
            logger.warning("Cannot create Meridian data without time periods and breaks")
            return None

        time_values = pd.to_datetime(time_periods["date"]).dt.strftime("%Y-%m-%d").tolist()
        geo_values = ["Israel"]

        program_types = sorted(set(breaks_data["program_type"].dropna().astype(str)) | set(PROGRAM_TYPES))
        media_channels = [
            f"{program_type}_{position}_{break_type}"
            for program_type in program_types
            for position in POSITIONS
            for break_type in BREAK_TYPES
        ]
        channel_index = {name: idx for idx, name in enumerate(media_channels)}

        kpi_data = np.ones((len(geo_values), len(time_values)), dtype=float)
        revenue_per_kpi_data = np.ones((len(geo_values), len(time_values)), dtype=float)
        media_data = np.zeros((len(geo_values), len(time_values), len(media_channels)), dtype=float)
        media_spend_data = np.zeros(len(media_channels), dtype=float)

        date_to_index = {date: idx for idx, date in enumerate(time_periods["date"])}
        for date, group in breaks_data.groupby("date"):
            date_key = pd.Timestamp(date).date()
            time_idx = date_to_index.get(date_key)
            if time_idx is None:
                continue

            retention = pd.to_numeric(group["avg_viewer_retention"], errors="coerce").mean()
            if pd.notna(retention):
                kpi_data[0, time_idx] = float(np.clip(retention, 0.01, 1.5))

            total_revenue = pd.to_numeric(group["total_revenue"], errors="coerce").sum()
            retention_sum = pd.to_numeric(group["avg_viewer_retention"], errors="coerce").sum()
            if retention_sum > 0:
                revenue_per_kpi_data[0, time_idx] = float(total_revenue / retention_sum)

            for _, row in group.iterrows():
                channel = f"{row['program_type']}_{row['position_category']}_{row['break_type']}"
                media_idx = channel_index.get(channel)
                if media_idx is None:
                    continue
                seconds = float(row.get("total_break_duration", 0) or 0)
                media_data[0, time_idx, media_idx] += seconds
                media_spend_data[media_idx] += seconds

        media_spend_data = np.where(media_spend_data > 0, media_spend_data, 1.0)

        control_variables = [
            "day_of_week_Monday",
            "day_of_week_Tuesday",
            "day_of_week_Wednesday",
            "day_of_week_Thursday",
            "day_of_week_Friday",
            "day_of_week_Saturday",
            "day_of_week_Sunday",
            "is_weekend",
            "avg_program_duration",
            "num_programs",
            "avg_program_tvr",
        ]
        controls_data = np.zeros((len(geo_values), len(time_values), len(control_variables)), dtype=float)

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for time_idx, row in time_periods.iterrows():
            day_name = row["day_of_week"]
            if day_name in day_order:
                controls_data[0, time_idx, day_order.index(day_name)] = 1.0
            controls_data[0, time_idx, 7] = 1.0 if bool(row["is_weekend"]) else 0.0

            if not programs_data.empty:
                date_programs = programs_data[programs_data["date"] == row["date"]]
                if not date_programs.empty:
                    controls_data[0, time_idx, 8] = pd.to_numeric(
                        date_programs["avg_program_duration"], errors="coerce"
                    ).mean()
                    controls_data[0, time_idx, 9] = pd.to_numeric(
                        date_programs["num_programs"], errors="coerce"
                    ).sum()
                    controls_data[0, time_idx, 10] = pd.to_numeric(
                        date_programs["avg_program_tvr"], errors="coerce"
                    ).mean()

        controls_data = np.nan_to_num(controls_data, nan=0.0)

        return input_data.InputData(
            kpi=xr.DataArray(kpi_data, dims=["geo", "time"], coords={"geo": geo_values, "time": time_values}, name="kpi"),
            kpi_type="non_revenue",
            controls=xr.DataArray(
                controls_data,
                dims=["geo", "time", "control_variable"],
                coords={"geo": geo_values, "time": time_values, "control_variable": control_variables},
                name="controls",
            ),
            population=xr.DataArray(np.array([1_000_000]), dims=["geo"], coords={"geo": geo_values}, name="population"),
            revenue_per_kpi=xr.DataArray(
                revenue_per_kpi_data,
                dims=["geo", "time"],
                coords={"geo": geo_values, "time": time_values},
                name="revenue_per_kpi",
            ),
            media=xr.DataArray(
                media_data,
                dims=["geo", "media_time", "media_channel"],
                coords={"geo": geo_values, "media_time": time_values, "media_channel": media_channels},
                name="media",
            ),
            media_spend=xr.DataArray(
                media_spend_data,
                dims=["media_channel"],
                coords={"media_channel": media_channels},
                name="media_spend",
            ),
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = argparse.ArgumentParser(description="Validate and transform TV break data")
    parser.add_argument("--dayparts", type=Path, default=Path("data/Dayparts.csv"))
    parser.add_argument("--programmes", type=Path, default=Path("data/Programmes.csv"))
    parser.add_argument("--spots", type=Path, default=Path("data/Spots.csv"))
    args = parser.parse_args()

    transformer = TVBreakDataTransformer(args.dayparts, args.programmes, args.spots)
    result = transformer.transform_data()
    aggregated = result["aggregated_data"]
    logger.info(
        "Transformed %s breaks, %s programme groups, Meridian data=%s",
        len(aggregated["breaks"]),
        len(aggregated["programs"]),
        result["meridian_data"] is not None,
    )


if __name__ == "__main__":
    main()
