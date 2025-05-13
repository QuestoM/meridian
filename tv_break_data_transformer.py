import pandas as pd
import numpy as np
import datetime as dt
import xarray as xr
from meridian.data import load
from meridian.data import input_data
import logging
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)

class TVBreakDataTransformer:
    def __init__(self, dayparts_path, programmes_path, spots_path, time_freq='D'):
        """Initialize with paths to data files."""
        self.dayparts_path = dayparts_path
        self.programmes_path = programmes_path
        self.spots_path = spots_path
        self.time_freq = time_freq  # 'D' = daily (default)

    def transform_data(self):
        """Transform TV commercial break data into Meridian-compatible format."""
        # Load raw data
        dayparts_df = pd.read_excel(self.dayparts_path)
        programmes_df = pd.read_excel(self.programmes_path)
        spots_df = pd.read_excel(self.spots_path)

        # Clean and preprocess data
        dayparts_clean = self._preprocess_dayparts(dayparts_df)
        programmes_clean = self._preprocess_programmes(programmes_df)
        spots_clean = self._preprocess_spots(spots_df)

        # Identify commercial breaks
        breaks_df = self._identify_commercial_breaks(spots_clean)

        # Match breaks with programs
        breaks_with_programs = self._match_breaks_with_programs(breaks_df, programmes_clean)

        # Calculate viewership impact metrics
        breaks_with_metrics = self._calculate_viewership_metrics(
            breaks_with_programs,
            dayparts_clean
        )

        # Calculate revenue metrics
        breaks_with_revenue = self._calculate_revenue_metrics(breaks_with_metrics)

        # Create aggregated data by time period
        aggregated_data = self._create_aggregated_data(
            breaks_with_revenue,
            programmes_clean,
            dayparts_clean
        )

        # Transform into Meridian input format
        meridian_data = self._create_meridian_data(aggregated_data)

        return {
            'raw_data': {
                'dayparts': dayparts_clean,
                'programmes': programmes_clean,
                'spots': spots_clean,
                'breaks': breaks_with_revenue
            },
            'aggregated_data': aggregated_data,
            'meridian_data': meridian_data
        }

    def _preprocess_dayparts(self, dayparts_df):
        """Clean and preprocess dayparts data."""
        # Clean up column names
        dayparts_df.columns = [col if not pd.isna(col) else f"col_{i}"
                              for i, col in enumerate(dayparts_df.columns)]

        # Get channel columns (assuming they contain 'TVR' in the column name)
        channel_cols = [col for col in dayparts_df.columns if str(col).find('TVR') >= 0]

        # Create a clean dataframe
        clean_df = pd.DataFrame()
        # Handle both original XLSX ('Dates'/'Timebands') and already processed CSV ('date'/'time')
        if 'Dates' in dayparts_df.columns and 'Timebands' in dayparts_df.columns:
        clean_df['date'] = dayparts_df['Dates']
        clean_df['time'] = dayparts_df['Timebands']
        elif 'date' in dayparts_df.columns and 'time' in dayparts_df.columns:
            clean_df['date'] = dayparts_df['date']
            clean_df['time'] = dayparts_df['time']
        else:
            raise ValueError("Dayparts file missing expected columns (Dates/Timebands or date/time)")

        # Standardise date column to python date objects for consistency
        try:
            # Force day-first as files use DD/MM/YYYY, strip whitespace first
            clean_df['date'] = (
                clean_df['date']
                .astype(str)
                .str.strip()
                .pipe(pd.to_datetime, format='%d/%m/%Y', dayfirst=True, errors='coerce')
                .dt.date
            )
        except Exception as e:
            logger.warning(f"Failed to coerce dayparts 'date' column to datetime: {e}")

        # Add channel columns
        for i, channel in enumerate(['עכשיו 14', 'קשת 12', 'רשת 13', 'כאן 11']):
            if i < len(channel_cols):
                clean_df[channel] = dayparts_df[channel_cols[i]]
            else:
                # Try to use direct columns if they exist (in pre-processed CSV)
                if channel in dayparts_df.columns:
                    clean_df[channel] = dayparts_df[channel]
            else:
                clean_df[channel] = np.nan

        # Convert date/time to datetime
        if 'datetime' in dayparts_df.columns:
            clean_df['datetime'] = pd.to_datetime(dayparts_df['datetime'])
        else:
            # date is already a python date; cast to str for concatenation
        clean_df['datetime'] = pd.to_datetime(
            clean_df['date'].astype(str) + ' ' + clean_df['time'].astype(str),
                format='%Y-%m-%d %H:%M',  # date now ISO after .dt.date -> str
            errors='coerce'
        )

        # Filter out rows with invalid datetime
        clean_df = clean_df.dropna(subset=['datetime'])

        return clean_df

    def _preprocess_programmes(self, programmes_df):
        """Clean and preprocess programmes data."""
        # Ensure datetime columns are correctly parsed
        for col in ['Start time', 'End time']:
            if col in programmes_df.columns:
                programmes_df[col] = pd.to_datetime(programmes_df[col], errors='coerce')

        # Calculate program duration if not already present
        if 'Duration' not in programmes_df.columns:
            programmes_df['Duration'] = (
                programmes_df['End time'] - programmes_df['Start time']
            ).dt.total_seconds()

        # Convert date string to datetime
        if 'Date' in programmes_df.columns:
            try:
                programmes_df['Date'] = pd.to_datetime(programmes_df['Date'].astype(str).str.strip(),
                                                     format='%d/%m/%Y', dayfirst=True, errors='coerce')
            except Exception:
                programmes_df['Date'] = pd.to_datetime(programmes_df['Date'], errors='coerce')

            # Keep only date part for aggregation consistency
            programmes_df['Date'] = programmes_df['Date'].dt.date

        # Categorize programs
        programmes_df['program_type'] = self._categorize_programs(programmes_df)

        # Flag prime time programs (18:00-23:00)
        if 'Start time' in programmes_df.columns:
            programmes_df['prime_time'] = (
                (programmes_df['Start time'].dt.hour >= 18) &
                (programmes_df['Start time'].dt.hour < 23)
            )

        return programmes_df

    def _preprocess_spots(self, spots_df):
        """Clean and preprocess spots data."""
        # Ensure datetime columns are correctly parsed
        if 'Start time' in spots_df.columns:
            spots_df['Start time'] = pd.to_datetime(spots_df['Start time'].astype(str).str.strip(), errors='coerce', dayfirst=True)

        # Convert duration to seconds if it's not already
        if 'Duration' in spots_df.columns:
            # Check if Duration is already in seconds or needs conversion
            if spots_df['Duration'].dtype == 'object':
                spots_df['Duration'] = spots_df['Duration'].str.strip().astype(float)

        # Convert date string to datetime
        if 'Date' in spots_df.columns:
            spots_df['Date'] = pd.to_datetime(
                spots_df['Date'].astype(str).str.strip(),
                format='%d/%m/%Y',
                dayfirst=True,
                errors='coerce'
            )

        # Keep date part only for consistency
        if 'Date' in spots_df.columns:
            spots_df['Date'] = spots_df['Date'].dt.date

        # Calculate end time
        spots_df['End time'] = spots_df['Start time'] + pd.to_timedelta(spots_df['Duration'], unit='s')

        return spots_df

    def _categorize_programs(self, programmes_df):
        """Categorize programs based on title keywords."""
        # This is a simplified version - would need enhancement for a real implementation
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

        return programmes_df['Title'].apply(categorize)

    def _identify_commercial_breaks(self, spots_df):
        """Identify commercial breaks from spots data."""
        # Sort spots by channel and time
        spots_df = spots_df.sort_values(['Channel', 'Start time'])

        # Initialize empty breaks dataframe
        breaks = []

        # Group spots by channel
        for channel, channel_spots in spots_df.groupby('Channel'):
            # Initialize variables for tracking breaks
            current_break = []

            # Iterate through spots chronologically
            for i, spot in channel_spots.iterrows():
                if not current_break:
                    # Start a new break
                    current_break = [spot]
                else:
                    # Check if this spot is part of the current break
                    last_spot = current_break[-1]
                    time_between = (spot['Start time'] - last_spot['End time']).total_seconds()

                    # If spots are close together (< 15 seconds), consider them part of the same break
                    if time_between <= 15:
                        current_break.append(spot)
                    else:
                        # End the current break and start a new one
                        if len(current_break) > 1:
                            break_start = current_break[0]['Start time']
                            break_end = current_break[-1]['End time']
                            break_duration = (break_end - break_start).total_seconds()

                            breaks.append({
                                'channel': channel,
                                'break_start': break_start,
                                'break_end': break_end,
                                'break_duration': break_duration,
                                'num_spots': len(current_break),
                                'spots': [s['Campaign'] for s in current_break],
                                'spot_ids': [s.name for s in current_break],
                                'avg_tvr': np.mean([s['TVR'] for s in current_break if 'TVR' in s])
                            })

                        # Start a new break with the current spot
                        current_break = [spot]

            # Don't forget the last break
            if len(current_break) > 1:
                break_start = current_break[0]['Start time']
                break_end = current_break[-1]['End time']
                break_duration = (break_end - break_start).total_seconds()

                breaks.append({
                    'channel': channel,
                    'break_start': break_start,
                    'break_end': break_end,
                    'break_duration': break_duration,
                    'num_spots': len(current_break),
                    'spots': [s['Campaign'] for s in current_break],
                    'spot_ids': [s.name for s in current_break],
                    'avg_tvr': np.mean([s['TVR'] for s in current_break if 'TVR' in s])
                })

        # Convert to DataFrame
        breaks_df = pd.DataFrame(breaks)

        # Add break type classification
        if not breaks_df.empty:
            breaks_df['break_type'] = breaks_df['break_duration'].apply(
                lambda x: 'Short' if x < 60 else ('Medium' if x < 120 else 'Long')
            )

        return breaks_df

    def _match_breaks_with_programs(self, breaks_df, programmes_df):
        """Match commercial breaks with their containing programs."""
        if breaks_df.empty:
            return breaks_df

        # Create copy to avoid modifying the original
        breaks_with_programs = breaks_df.copy()

        # Initialize columns for program info
        breaks_with_programs['program_title'] = None
        breaks_with_programs['program_type'] = None
        breaks_with_programs['program_duration'] = None
        breaks_with_programs['seconds_into_program'] = None
        breaks_with_programs['relative_position'] = None
        breaks_with_programs['program_tvr'] = None
        breaks_with_programs['prime_time'] = False

        # For each break, find the containing program
        for idx, break_info in breaks_with_programs.iterrows():
            matching_programs = programmes_df[
                (programmes_df['Channel'] == break_info['channel']) &
                (programmes_df['Start time'] <= break_info['break_start']) &
                (programmes_df['End time'] >= break_info['break_end'])
            ]

            if not matching_programs.empty:
                program = matching_programs.iloc[0]
                breaks_with_programs.at[idx, 'program_title'] = program['Title']
                breaks_with_programs.at[idx, 'program_type'] = program['program_type']
                breaks_with_programs.at[idx, 'program_duration'] = program['Duration']
                breaks_with_programs.at[idx, 'program_tvr'] = program['TVR'] if 'TVR' in program else None

                # Calculate position in program
                seconds_into_program = (break_info['break_start'] - program['Start time']).total_seconds()
                breaks_with_programs.at[idx, 'seconds_into_program'] = seconds_into_program

                if program['Duration'] > 0:
                    relative_position = seconds_into_program / program['Duration']
                    breaks_with_programs.at[idx, 'relative_position'] = relative_position

                    # Categorize break position
                    if relative_position <= 0.33:
                        breaks_with_programs.at[idx, 'position_category'] = 'Early'
                    elif relative_position <= 0.66:
                        breaks_with_programs.at[idx, 'position_category'] = 'Middle'
                    else:
                        breaks_with_programs.at[idx, 'position_category'] = 'Late'

                # Set prime time flag
                if 'prime_time' in program:
                    breaks_with_programs.at[idx, 'prime_time'] = program['prime_time']

        return breaks_with_programs

    def _calculate_viewership_metrics(self, breaks_df, dayparts_df):
        """Calculate viewership metrics for each break."""
        if breaks_df.empty:
            return breaks_df

        # Create copy to avoid modifying the original
        breaks_with_metrics = breaks_df.copy()

        # Initialize columns for viewership metrics
        breaks_with_metrics['pre_break_tvr'] = None
        breaks_with_metrics['during_break_tvr'] = breaks_with_metrics['avg_tvr']
        breaks_with_metrics['post_break_tvr'] = None
        breaks_with_metrics['viewer_retention'] = None
        breaks_with_metrics['viewer_recovery'] = None

        # For each break, calculate viewership metrics
        for idx, break_info in breaks_with_metrics.iterrows():
            # Get channel column name
            channel_col = break_info['channel']

            # Get pre-break viewership (5 minutes before)
            pre_break_start = break_info['break_start'] - pd.Timedelta(minutes=5)
            pre_break_data = dayparts_df[
                (dayparts_df['datetime'] >= pre_break_start) &
                (dayparts_df['datetime'] < break_info['break_start'])
            ]

            if not pre_break_data.empty and channel_col in pre_break_data.columns:
                pre_break_tvr = pre_break_data[channel_col].mean()
                breaks_with_metrics.at[idx, 'pre_break_tvr'] = pre_break_tvr

            # Get post-break viewership (5 minutes after)
            post_break_end = break_info['break_end'] + pd.Timedelta(minutes=5)
            post_break_data = dayparts_df[
                (dayparts_df['datetime'] > break_info['break_end']) &
                (dayparts_df['datetime'] <= post_break_end)
            ]

            if not post_break_data.empty and channel_col in post_break_data.columns:
                post_break_tvr = post_break_data[channel_col].mean()
                breaks_with_metrics.at[idx, 'post_break_tvr'] = post_break_tvr

            # Calculate viewer retention
            if (breaks_with_metrics.at[idx, 'pre_break_tvr'] is not None and
                breaks_with_metrics.at[idx, 'during_break_tvr'] is not None and
                breaks_with_metrics.at[idx, 'pre_break_tvr'] > 0):
                retention = breaks_with_metrics.at[idx, 'during_break_tvr'] / breaks_with_metrics.at[idx, 'pre_break_tvr']
                breaks_with_metrics.at[idx, 'viewer_retention'] = retention

            # Calculate viewer recovery
            if (breaks_with_metrics.at[idx, 'during_break_tvr'] is not None and
                breaks_with_metrics.at[idx, 'post_break_tvr'] is not None and
                breaks_with_metrics.at[idx, 'during_break_tvr'] > 0):
                recovery = breaks_with_metrics.at[idx, 'post_break_tvr'] / breaks_with_metrics.at[idx, 'during_break_tvr']
                breaks_with_metrics.at[idx, 'viewer_recovery'] = recovery

        return breaks_with_metrics

    def _calculate_revenue_metrics(self, breaks_df):
        """Calculate revenue metrics for each break."""
        if breaks_df.empty:
            return breaks_df

        # Create copy to avoid modifying the original
        breaks_with_revenue = breaks_df.copy()

        # For this example, we'll use simplified revenue calculations
        # In a real implementation, we would use actual ad rates and inventory data

        # Define some assumptions for revenue calculation
        base_rate_per_rating_point = 1000  # Base cost per rating point (in currency units)
        prime_time_multiplier = 1.5  # Prime time costs 50% more

        # Calculate revenue for each break
        for idx, break_info in breaks_with_revenue.iterrows():
            tvr = break_info['during_break_tvr'] if break_info['during_break_tvr'] is not None else 1.0

            # Apply prime time multiplier if applicable
            rate_multiplier = prime_time_multiplier if break_info['prime_time'] else 1.0

            # Calculate revenue
            revenue = tvr * base_rate_per_rating_point * rate_multiplier * (break_info['break_duration'] / 60)
            breaks_with_revenue.at[idx, 'estimated_revenue'] = revenue

            # Calculate revenue per minute
            if break_info['break_duration'] > 0:
                revenue_per_minute = revenue / (break_info['break_duration'] / 60)
                breaks_with_revenue.at[idx, 'revenue_per_minute'] = revenue_per_minute

            # Calculate cost per viewer
            if tvr > 0:
                cost_per_viewer = revenue / tvr
                breaks_with_revenue.at[idx, 'cost_per_viewer'] = cost_per_viewer

        return breaks_with_revenue

    def _create_aggregated_data(self, breaks_df, programmes_df, dayparts_df):
        """Create aggregated data by time period."""
        # Get all available dates from the data
        available_dates = []

        if not breaks_df.empty and 'break_start' in breaks_df.columns:
            available_dates.extend(pd.to_datetime(breaks_df['break_start']).dt.date.unique())

        if 'Date' in programmes_df.columns:
            try:
                prog_dates = pd.to_datetime(programmes_df['Date']).dt.date.unique()
                available_dates.extend(prog_dates)
            except:
                logger.warning("Could not parse Date column in programmes_df")

        if 'date' in dayparts_df.columns:
            try:
                dayparts_dates = pd.to_datetime(dayparts_df['date']).dt.date.unique()
                available_dates.extend(dayparts_dates)
            except:
                logger.warning("Could not parse date column in dayparts_df")

        # Remove unrealistic dates (<2010-01-01 or >2100-01-01) then unique+sort
        cleaned_dates = [d for d in available_dates if d is not None and 2010 <= d.year <= 2100]

        # Remove duplicates and sort
        available_dates = sorted(list(set(cleaned_dates)))

        if len(available_dates) < 2:
            logger.warning(f"Only {len(available_dates)} unique dates found. Creating at least 2 days of data.")
            if len(available_dates) == 1:
                start_date = available_dates[0]
                end_date = start_date + pd.Timedelta(days=1)
        else:
                # Fallback to a default date range if no dates found
                start_date = pd.Timestamp('2023-01-01').date()
                end_date = pd.Timestamp('2023-01-02').date()
        else:
            start_date = min(available_dates)
            end_date = max(available_dates)

        # Generate a complete date range (ensuring we have data for every day)
        date_range = pd.date_range(start=start_date, end=end_date, freq=self.time_freq)

        # Create time periods dataframe
        time_periods = pd.DataFrame({'date': date_range.date})

        # Log time dimension information for diagnostics
        logger.info(f"Created time dimension with {len(time_periods)} periods from {start_date} to {end_date}")
        logger.info(f"Using time frequency: {self.time_freq}")

        # Add day of week
        time_periods['day_of_week'] = time_periods['date'].apply(lambda d: pd.Timestamp(d).day_name())
        time_periods['is_weekend'] = time_periods['day_of_week'].isin(['Saturday', 'Sunday'])

        # Group breaks by day and program type
        aggregated_breaks = []

        if not breaks_df.empty:
            # Add date to breaks for grouping
            breaks_df['date'] = pd.to_datetime(breaks_df['break_start'].dt.date)

            # Group by date, channel, and program type
            for (date, channel, program_type), group in breaks_df.groupby(['date', 'channel', 'program_type']):
                # Skip if program type is None
                if program_type is None:
                    continue

                aggregated_breaks.append({
                    'date': date,
                    'channel': channel,
                    'program_type': program_type,
                    'num_breaks': len(group),
                    'total_break_duration': group['break_duration'].sum(),
                    'avg_break_duration': group['break_duration'].mean(),
                    'total_spots': group['num_spots'].sum(),
                    'avg_viewer_retention': group['viewer_retention'].mean(),
                    'avg_viewer_recovery': group['viewer_recovery'].mean(),
                    'total_revenue': group['estimated_revenue'].sum(),
                    'avg_revenue_per_minute': group['revenue_per_minute'].mean() if 'revenue_per_minute' in group.columns else None
                })

        # Convert to DataFrame
        aggregated_breaks_df = pd.DataFrame(aggregated_breaks) if aggregated_breaks else pd.DataFrame()

        # Group programs by day and type
        aggregated_programs = []

        # Add date to programs for grouping
        programmes_df['date'] = pd.to_datetime(programmes_df['Date'])

        # Group by date, channel, and program type
        for (date, channel, program_type), group in programmes_df.groupby(['date', 'Channel', 'program_type']):
            # Skip if program type is None
            if program_type is None:
                continue

            aggregated_programs.append({
                'date': date,
                'channel': channel,
                'program_type': program_type,
                'num_programs': len(group),
                'total_program_duration': group['Duration'].sum(),
                'avg_program_duration': group['Duration'].mean(),
                'avg_program_tvr': group['TVR'].mean() if 'TVR' in group.columns else None,
                'num_prime_time_programs': group['prime_time'].sum() if 'prime_time' in group.columns else 0
            })

        # Convert to DataFrame
        aggregated_programs_df = pd.DataFrame(aggregated_programs) if aggregated_programs else pd.DataFrame()

        # Aggregate viewership by day and channel
        aggregated_viewership = []

        # Add date to dayparts for grouping
        try:
            # Try different formats depending on data
        dayparts_df['date'] = pd.to_datetime(dayparts_df['date'], format='%d/%m/%Y')
        except ValueError:
            # If the above fails, try ISO format
            dayparts_df['date'] = pd.to_datetime(dayparts_df['date'])

        # Get the channel columns (excluding metadata columns)
        channel_cols = [col for col in dayparts_df.columns
                        if col not in ['date', 'time', 'datetime', 'col_0']]

        # Group by date
        for date, group in dayparts_df.groupby('date'):
            for channel in channel_cols:
                if channel in group.columns:
                    # Calculate average viewership for the day
                    avg_viewership = group[channel].mean()

                    # Calculate peak viewership
                    peak_viewership = group[channel].max()

                    # Calculate prime time viewership (18:00-23:00)
                    prime_time_data = group[
                        (group['datetime'].dt.hour >= 18) &
                        (group['datetime'].dt.hour < 23)
                    ]

                    prime_time_viewership = prime_time_data[channel].mean() if not prime_time_data.empty else None

                    aggregated_viewership.append({
                        'date': date,
                        'channel': channel,
                        'avg_viewership': avg_viewership,
                        'peak_viewership': peak_viewership,
                        'prime_time_viewership': prime_time_viewership
                    })

        # Convert to DataFrame
        aggregated_viewership_df = pd.DataFrame(aggregated_viewership) if aggregated_viewership else pd.DataFrame()

        # Return all aggregated data
        return {
            'time_periods': time_periods,
            'breaks': aggregated_breaks_df,
            'programs': aggregated_programs_df,
            'viewership': aggregated_viewership_df
        }

    def _create_meridian_data(self, aggregated_data):
        """Transform aggregated data into Meridian's input format."""
        # Extract components
        time_periods = aggregated_data['time_periods']
        breaks_data = aggregated_data['breaks']
        programs_data = aggregated_data['programs']
        viewership_data = aggregated_data['viewership']

        # If we don't have enough data, return None
        if time_periods.empty or breaks_data.empty:
            return None

        # Create time dimension
        # Make sure we have at least 2 time periods with regular spacing
        if len(time_periods) < 2:
            # If we don't have enough dates, create a date range
            start_date = time_periods['date'].iloc[0] if not time_periods.empty else pd.Timestamp('2023-01-01')
            dates = pd.date_range(start=start_date, periods=7, freq='D')  # Create a week of dates
            time_values = dates.strftime('%Y-%m-%d').tolist()
        else:
            # Check if dates are regularly spaced
            dates = pd.to_datetime(time_periods['date'])
            date_diffs = dates.diff()[1:]  # Get differences between consecutive dates

            if date_diffs.nunique() > 1:
                # Dates are not regularly spaced, create a regular date range instead
                min_date = dates.min()
                max_date = dates.max()
                # Use the most common difference as frequency
                most_common_diff = date_diffs.value_counts().index[0]
                regular_dates = pd.date_range(start=min_date, end=max_date, freq=most_common_diff)
                time_values = regular_dates.strftime('%Y-%m-%d').tolist() if hasattr(regular_dates, 'strftime') else [d.strftime('%Y-%m-%d') for d in regular_dates]
            else:
                # Dates are already regularly spaced
                time_values = dates.dt.strftime('%Y-%m-%d').tolist()

        # Create geo dimension (national level for Israeli TV)
        geo_values = ['Israel']

        # Create media channels based on break types and positions
        # Combine program type, break duration, and position
        media_channels = []

        if not breaks_data.empty:
            # Get unique combinations of program type and break types
            program_types = sorted(breaks_data['program_type'].unique())

            for program_type in program_types:
                media_channels.extend([
                    f'{program_type}_Early_Short',
                    f'{program_type}_Early_Medium',
                    f'{program_type}_Early_Long',
                    f'{program_type}_Middle_Short',
                    f'{program_type}_Middle_Medium',
                    f'{program_type}_Middle_Long',
                    f'{program_type}_Late_Short',
                    f'{program_type}_Late_Medium',
                    f'{program_type}_Late_Long'
                ])

        # If no media channels were created, use some defaults
        if not media_channels:
            media_channels = [
                'Drama_Early_Short',
                'Drama_Middle_Medium',
                'Drama_Late_Short',
                'News_Early_Short',
                'News_Middle_Medium',
                'News_Late_Short'
            ]

        # Create control variables
        control_variables = [
            'day_of_week_Monday',
            'day_of_week_Tuesday',
            'day_of_week_Wednesday',
            'day_of_week_Thursday',
            'day_of_week_Friday',
            'day_of_week_Saturday',
            'day_of_week_Sunday',
            'is_weekend',
            'avg_program_duration',
            'num_programs',
            'avg_program_tvr'
        ]

        # Create KPI data (viewer retention)
        # For each time period and channel, calculate average viewer retention
        kpi_data = np.ones((len(geo_values), len(time_values)))

        if not breaks_data.empty and 'avg_viewer_retention' in breaks_data.columns:
            # Group by date
            for date_idx, date in enumerate(time_periods['date']):
                date_breaks = breaks_data[breaks_data['date'] == date]
                if not date_breaks.empty:
                    avg_retention = date_breaks['avg_viewer_retention'].mean()
                    if not pd.isna(avg_retention):
                        kpi_data[0, date_idx] = avg_retention

        # Create KPI DataArray
        kpi = xr.DataArray(
            kpi_data,
            dims=['geo', 'time'],
            coords={
                'geo': geo_values,
                'time': time_values
            },
            name='kpi'
        )

        # Create revenue per KPI (ad rates)
        # For each time period, calculate average revenue per rating point
        revenue_per_kpi_data = np.ones((len(geo_values), len(time_values)))

        if not breaks_data.empty and 'total_revenue' in breaks_data.columns and 'avg_viewer_retention' in breaks_data.columns:
            # Group by date
            for date_idx, date in enumerate(time_periods['date']):
                date_breaks = breaks_data[breaks_data['date'] == date]
                if not date_breaks.empty:
                    total_revenue = date_breaks['total_revenue'].sum()
                    total_retention = date_breaks['avg_viewer_retention'].sum()
                    if total_retention > 0:
                        revenue_per_rating = total_revenue / total_retention
                        revenue_per_kpi_data[0, date_idx] = revenue_per_rating

        # Create revenue per KPI DataArray
        revenue_per_kpi = xr.DataArray(
            revenue_per_kpi_data,
            dims=['geo', 'time'],
            coords={
                'geo': geo_values,
                'time': time_values
            },
            name='revenue_per_kpi'
        )

        # Create controls data
        controls_data = np.zeros((len(geo_values), len(time_values), len(control_variables)))

        # Fill controls data based on time periods and program data
        for time_idx, date in enumerate(time_periods['date']):
            # Day of week one-hot encoding
            day_of_week = time_periods.loc[time_periods['date'] == date, 'day_of_week'].iloc[0]
            for dow_idx, dow in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']):
                if day_of_week == dow:
                    controls_data[0, time_idx, dow_idx] = 1

            # Weekend flag
            is_weekend = time_periods.loc[time_periods['date'] == date, 'is_weekend'].iloc[0]
            controls_data[0, time_idx, 7] = 1 if is_weekend else 0

            # Program metrics
            if not programs_data.empty:
                date_programs = programs_data[programs_data['date'] == date]
                if not date_programs.empty:
                    # Average program duration
                    avg_duration = date_programs['avg_program_duration'].mean()
                    controls_data[0, time_idx, 8] = avg_duration if not pd.isna(avg_duration) else 0

                    # Number of programs
                    num_programs = date_programs['num_programs'].sum()
                    controls_data[0, time_idx, 9] = num_programs

                    # Average program TVR
                    avg_tvr = date_programs['avg_program_tvr'].mean()
                    controls_data[0, time_idx, 10] = avg_tvr if not pd.isna(avg_tvr) else 0

        # Create controls DataArray
        controls = xr.DataArray(
            controls_data,
            dims=['geo', 'time', 'control_variable'],
            coords={
                'geo': geo_values,
                'time': time_values,
                'control_variable': control_variables
            },
            name='controls'
        )

        # Create media data
        # For this example, we'll create simulated media data based on the break types
        # In a real implementation, we would use actual break data

        # Create media time dimension (same as time for simplicity)
        media_time_values = time_values

        # Initialize media data
        media_data = np.zeros((len(geo_values), len(media_time_values), len(media_channels)))

        # Fill media data based on break data
        if not breaks_data.empty:
            for time_idx, date in enumerate(time_periods['date']):
                date_breaks = breaks_data[breaks_data['date'] == date]

                for _, break_info in date_breaks.iterrows():
                    if pd.isna(break_info['program_type']):
                        continue

                    # Determine position category (default to Middle if missing)
                    position = break_info.get('position_category', 'Middle')

                    # Determine break type based on duration
                    duration = break_info['avg_break_duration']
                    if pd.isna(duration):
                        break_type = 'Medium'
                    elif duration < 60:
                        break_type = 'Short'
                    elif duration < 120:
                        break_type = 'Medium'
                    else:
                        break_type = 'Long'

                    # Construct the channel name
                    channel_name = f"{break_info['program_type']}_{position}_{break_type}"

                    # Find the channel index
                    if channel_name in media_channels:
                        channel_idx = media_channels.index(channel_name)

                        # Add break count to media data
                        media_data[0, time_idx, channel_idx] += break_info['num_breaks']

        # Create media DataArray
        media = xr.DataArray(
            media_data,
            dims=['geo', 'media_time', 'media_channel'],
            coords={
                'geo': geo_values,
                'media_time': media_time_values,
                'media_channel': media_channels
            },
            name='media'
        )

        # Create media spend data
        # For this example, we'll use a simplified model where spending is proportional to break duration
        media_spend_data = np.zeros(len(media_channels))

        if not breaks_data.empty:
            # For each media channel (break type), calculate total spend
            for channel_idx, channel_name in enumerate(media_channels):
                # Parse channel name
                parts = channel_name.split('_')
                if len(parts) == 3:
                    program_type, position, break_type = parts

                    # Filter breaks by program type and position
                    matching_breaks = breaks_data[
                        (breaks_data['program_type'] == program_type)
                    ]

                    if not matching_breaks.empty:
                        # Calculate average spend for this type of break
                        # Check if position_category column exists
                        if 'position_category' in matching_breaks.columns:
                            # Filter by position, treating NaN values as 'Middle'
                            matching_breaks_with_position = matching_breaks[
                                (matching_breaks['position_category'] == position) |
                                (pd.isna(matching_breaks['position_category']) & (position == 'Middle'))
                            ]
                        else:
                            # If column doesn't exist at all, only include rows if position is 'Middle'
                            if position == 'Middle':
                                matching_breaks_with_position = matching_breaks
                            else:
                                matching_breaks_with_position = matching_breaks.iloc[0:0]  # Empty DataFrame with same structure

                        if not matching_breaks_with_position.empty:
                            # Filter by break type
                            if break_type == 'Short':
                                matching_breaks_final = matching_breaks_with_position[
                                    matching_breaks_with_position['avg_break_duration'] < 60
                                ]
                            elif break_type == 'Medium':
                                matching_breaks_final = matching_breaks_with_position[
                                    (matching_breaks_with_position['avg_break_duration'] >= 60) &
                                    (matching_breaks_with_position['avg_break_duration'] < 120)
                                ]
                            else:  # Long
                                matching_breaks_final = matching_breaks_with_position[
                                    matching_breaks_with_position['avg_break_duration'] >= 120
                                ]

                            if not matching_breaks_final.empty and 'total_revenue' in matching_breaks_final.columns:
                                total_revenue = matching_breaks_final['total_revenue'].sum()
                                media_spend_data[channel_idx] = total_revenue

        # If no data was filled, use default values
        if np.sum(media_spend_data) == 0:
            media_spend_data = np.ones(len(media_channels)) * 1000  # Default spend

        # Create media spend DataArray
        media_spend = xr.DataArray(
            media_spend_data,
            dims=['media_channel'],
            coords={
                'media_channel': media_channels
            },
            name='media_spend'
        )

        # Create population data (total potential viewers)
        # For this example, we'll use a constant population
        population_data = np.array([1000000])  # 1 million potential viewers

        # Create population DataArray
        population = xr.DataArray(
            population_data,
            dims=['geo'],
            coords={
                'geo': geo_values
            },
            name='population'
        )

        # Create Meridian InputData object
        meridian_data = input_data.InputData(
            kpi=kpi,
            kpi_type='non_revenue',
            controls=controls,
            population=population,
            revenue_per_kpi=revenue_per_kpi,
            media=media,
            media_spend=media_spend
        )

        return meridian_data


# Example usage:
# transformer = TVBreakDataTransformer('dayparts.xlsx', 'programmes.xlsx', 'spots.xlsx')
# result = transformer.transform_data()
# meridian_data = result['meridian_data']

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")
    parser = argparse.ArgumentParser(description="Train TVBreakModel and save frozen posterior")
    parser.add_argument("--enriched_dir", type=Path, default=Path("data/enriched"))
    parser.add_argument("--output_dir", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_prior", type=int, default=5000, help="Number of prior samples")
    parser.add_argument("--n_keep", type=int, default=1000, help="Number of posterior samples to keep")
    parser.add_argument("--n_chains", type=int, default=8, help="Number of MCMC chains")
    parser.add_argument("--n_adapt", type=int, default=2000, help="Number of adaptation steps")
    parser.add_argument("--n_burnin", type=int, default=2000, help="Number of burn-in steps")
    parser.add_argument("--time_freq", type=str, default="D", help="Time frequency for data aggregation ('D'=daily)")
    args = parser.parse_args()

    dayparts_csv = args.enriched_dir / "Dayparts.csv"
    programmes_csv = args.enriched_dir / "Programmes.csv"
    spots_csv = args.enriched_dir / "Spots.csv"

    build_and_train(dayparts_csv, programmes_csv, spots_csv, args.output_dir,
                   seed=args.seed,
                   n_prior=args.n_prior,
                   n_chains=args.n_chains,
                   n_adapt=args.n_adapt,
                   n_burnin=args.n_burnin,
                   n_keep=args.n_keep,
                   time_freq=args.time_freq)

def build_and_train(dayparts_csv: Path, programmes_csv: Path, spots_csv: Path, output_dir: Path,
                    seed: int = 42, n_prior: int = 500, n_chains: int = 4, n_adapt: int = 500,
                    n_burnin: int = 500, n_keep: int = 1000, time_freq: str = 'D'):
    """Load enriched CSVs, create InputData via legacy transformer (fast), fit TVBreakModel and save posterior."""

    logger.info(f"PARAMETERS: n_prior={n_prior}, n_chains={n_chains}, n_adapt={n_adapt}, n_burnin={n_burnin}, n_keep={n_keep}, time_freq={time_freq}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # The legacy TVBreakDataTransformer expects Excel; patch read_excel so it can read CSV too
    original_read_excel = pd.read_excel

    def adaptive_read_excel(path, *args, **kwargs):  # type: ignore[override]
        path = Path(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, *args, **kwargs)
        return original_read_excel(path, *args, **kwargs)

    pd.read_excel = adaptive_read_excel  # type: ignore

    try:
        transformer = TVBreakDataTransformer(dayparts_csv, programmes_csv, spots_csv, time_freq=time_freq)
        data_result = transformer.transform_data()
    finally:
        # Always restore original pandas read_excel
        pd.read_excel = original_read_excel

    # ... rest of the function remains unchanged ...
