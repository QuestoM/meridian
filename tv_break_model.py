# tv_break_model.py
# FINAL VERSION incorporating fixes for AttributeError and ensuring proper initialization

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import xarray as xr
import pandas as pd
import dataclasses
from dataclasses import field # Keep field import just in case it's used elsewhere indirectly
from typing import Any, Dict, List, Optional
import logging # Added for logging warnings

# Assuming these imports exist and work
from meridian import constants
from meridian.model import model # Assuming model.Meridian and model.NotFittedModelError exist
from meridian.model import spec # Explicitly import spec for inheritance
from meridian.model import prior_distribution
# Analyzer is no longer needed for the specific task but might be used elsewhere
from meridian.analysis import analyzer
from meridian.analysis import optimizer

# Get a logger instance (optional, but good practice)
# Use the name of the main script if logger is configured there, or __name__
# Assuming logger is configured in the main script 'tv_break_optimizer'
logger = logging.getLogger('tv_break_optimizer')
# If not configured elsewhere, use: logger = logging.getLogger(__name__)

# === START: Updated TVBreakModelSpec class ===

# Make the child class a frozen dataclass too
@dataclasses.dataclass(frozen=True)
class TVBreakModelSpec(spec.ModelSpec):
    """Model specification optimized for TV commercial break analysis.

    This extends the standard Meridian ModelSpec with parameters
    specifically tuned for analyzing the impact of commercial breaks.
    """
    # Declare the new field explicitly. Dataclass handles its initialization.
    roi_calibration_period: Optional[Any] = None

    # --- Modified __init__ method ---
    def __init__(self, **kwargs):
        # --- Logic to prepare arguments BEFORE calling super().__init__ ---

        # a) Define the TV-specific priors first
        tv_priors = prior_distribution.PriorDistribution(
            alpha_m=tfp.distributions.LogNormal(0.1, 0.5, name=constants.ALPHA_M),
            roi_m=tfp.distributions.LogNormal(0.3, 0.7, name=constants.ROI_M),
            ec_m=tfp.distributions.HalfNormal(2.0, name=constants.EC_M),
            sigma=tfp.distributions.LogNormal(-1.5, 0.7, name=constants.SIGMA),
        )

        # b) Set up the dictionary of ALL default arguments for the parent
        tv_kwargs = {
            # Original TV defaults
            'max_lag': 1,
            'knots': 5,
            'hill_before_adstock': False,
            # Use constants.MEDIA_EFFECTS_NORMAL based on user warning log
            'media_effects_dist': constants.MEDIA_EFFECTS_NORMAL,
            'paid_media_prior_type': constants.PAID_MEDIA_PRIOR_TYPE_ROI,

            # ADD the arguments for the fields we need initialized:
            # Provide the default value for the new field (uses class default unless overridden)
            # Note: For frozen dataclasses, modifying self after init isn't allowed.
            # We rely on the default or the value passed in kwargs.
            # 'roi_calibration_period': self.roi_calibration_period, # This doesn't work with frozen dataclass init logic
            # Instead, let kwargs override the default directly if provided
            'prior': tv_priors, # Provide the custom priors object
        }
        # If roi_calibration_period needs a default mechanism with frozen=True,
        # it might require a factory or post_init processing if the default depends on self.
        # For now, assume kwargs handles it or the default None is sufficient initially.


        # c) Allow user-provided kwargs to override ANY of the defaults
        tv_kwargs.update(kwargs)

        # d) Call super().__init__ ONCE with the combined arguments.
        super().__init__(**tv_kwargs)

# === END: Updated TVBreakModelSpec class ===


# --- TVBreakModel Class ---

class TVBreakModel(model.Meridian):
    """Specialized model for TV commercial break optimization.

    This model extends Meridian to include specialized methods for
    analyzing and optimizing TV commercial breaks.
    """
    def __init__(self, input_data, model_spec=None, inference_data=None):
        # Use TV-specific model spec if none provided
        model_spec = model_spec or TVBreakModelSpec() # This now uses the updated class

        # Initialize parent class
        super().__init__(input_data, model_spec, inference_data)

        # Additional TV-specific attributes
        self.viewer_sensitivity = None  # Will hold viewer sensitivity calculations

    def analyze_break_impact(self):
        """Analyze how different break attributes impact viewership retention.

        Returns:
            dict: Analysis results containing impact factors by break attributes.
        """
        # Ensure model has been fitted
        if constants.POSTERIOR not in self.inference_data.groups():
            raise model.NotFittedModelError(
                "Analyzing break impact requires a fitted model."
            )

        # --- MODIFIED SECTION START ---
        # Get coefficients directly from inference_data posterior
        # Use the string name directly as constant was incorrect
        param_name = 'beta_m'
        if param_name not in self.inference_data.posterior:
             # Check if maybe the constant name string exists after all?
             param_name_const = constants.BETA_MEDIA_PARAM_NAME if hasattr(constants, 'BETA_MEDIA_PARAM_NAME') else 'beta_media' # Safer check
             if param_name_const in self.inference_data.posterior:
                 param_name = param_name_const
             else:
                 available_vars = list(self.inference_data.posterior.data_vars.keys())
                 raise KeyError(
                     f"Could not find media coefficients ('{param_name}' or potential constant name) "
                     f"in posterior samples. Available vars: {available_vars}"
                 )

        posterior_coefs = self.inference_data.posterior[param_name]

        # Calculate mean across chains and draws.
        # Use try-except for dimension names
        try:
            mean_coefs_xr = posterior_coefs.mean(dim=(constants.CHAIN, constants.DRAW))
        except ValueError as e:
             logger.debug(f"Failed using constant dim names ('{constants.CHAIN}', '{constants.DRAW}') for mean calculation: {e}. Trying default names ('chain', 'draw').")
             try:
                 mean_coefs_xr = posterior_coefs.mean(dim=('chain', 'draw'))
             except ValueError as e2:
                  raise ValueError(f"Could not calculate mean coefficients across chains and draws. Original error: {e}, Second attempt error: {e2}")


        # Squeeze out geo dimension if it exists and is size 1 (for national models)
        geo_dim_const = constants.GEO if hasattr(constants, 'GEO') else 'geo'
        if geo_dim_const in mean_coefs_xr.dims and mean_coefs_xr.dims[geo_dim_const] == 1:
            mean_coefs_xr = mean_coefs_xr.squeeze(geo_dim_const, drop=True)


        # Verify the coordinate name for media channels
        channel_coord_const = constants.MEDIA_CHANNEL if hasattr(constants, 'MEDIA_CHANNEL') else 'media_channel'
        if channel_coord_const in mean_coefs_xr.coords:
             media_channel_coord_name = channel_coord_const
        elif 'media_channel' in mean_coefs_xr.coords: # Try lowercase alternative explicitly
             media_channel_coord_name = 'media_channel'
        else:
             available_coords = list(mean_coefs_xr.coords.keys())
             raise KeyError(
                 f"Coordinate '{channel_coord_const}' or 'media_channel' "
                 f"not found in calculated mean coefficients. Available coords: {available_coords}"
             )

        channel_coefs_mean = mean_coefs_xr # This is now an xarray DataArray indexed by channel
        # --- MODIFIED SECTION END ---


        # Get program type factors (split from channel names)
        program_types = set()
        program_type_impacts = {}

        # Use the correct coordinate name identified above
        for channel in channel_coefs_mean[media_channel_coord_name].values:
            parts = str(channel).split('_')
            if len(parts) >= 3:
                program_type = parts[0]
                position = parts[1] # Keep original case for dict keys initially
                break_type = parts[2] # Keep original case for dict keys initially

                program_types.add(program_type)

                # Extract the mean coefficient using the identified coordinate name
                try:
                   # .item() extracts the scalar value from the 0-d array
                   mean_coef = float(channel_coefs_mean.sel({media_channel_coord_name: channel}).item())
                except IndexError:
                     logger.warning(f"Could not extract coefficient for channel: {channel}. Setting to 0.")
                     mean_coef = 0.0
                except Exception as e:
                     logger.error(f"Error selecting coefficient for channel {channel}: {e}")
                     mean_coef = 0.0

                # Store by program type
                if program_type not in program_type_impacts:
                    program_type_impacts[program_type] = {
                        'early_short': None, 'early_medium': None, 'early_long': None,
                        'middle_short': None, 'middle_medium': None, 'middle_long': None,
                        'late_short': None, 'late_medium': None, 'late_long': None,
                        'total_impact': 0.0, 'count': 0 # Initialize total_impact as float
                    }

                # Store in appropriate category (using lowercase for consistency)
                key = f"{position.lower()}_{break_type.lower()}"
                if key in program_type_impacts[program_type]:
                    program_type_impacts[program_type][key] = mean_coef
                    # Ensure impact is float before adding
                    program_type_impacts[program_type]['total_impact'] += float(mean_coef)
                    program_type_impacts[program_type]['count'] += 1

        # Calculate average impacts by position and length
        position_impacts = {
            'early': {'total': 0.0, 'count': 0, 'average': None}, # Use float for total
            'middle': {'total': 0.0, 'count': 0, 'average': None},
            'late': {'total': 0.0, 'count': 0, 'average': None}
        }

        length_impacts = {
            'short': {'total': 0.0, 'count': 0, 'average': None}, # Use float for total
            'medium': {'total': 0.0, 'count': 0, 'average': None},
            'long': {'total': 0.0, 'count': 0, 'average': None}
        }

        # Populate position and length impacts
        for program_type, impacts in program_type_impacts.items():
            for key, value in impacts.items():
                if key not in ['total_impact', 'count', 'average'] and value is not None: # Added 'average' check
                    try:
                        position, length = key.split('_')
                        value_float = float(value) # Ensure float
                        if position in position_impacts:
                           position_impacts[position]['total'] += value_float
                           position_impacts[position]['count'] += 1
                        if length in length_impacts:
                           length_impacts[length]['total'] += value_float
                           length_impacts[length]['count'] += 1
                    except ValueError:
                        logger.warning(f"Could not parse key '{key}' or convert value '{value}' to float in analyze_break_impact")
                    except Exception as e:
                        logger.warning(f"Error processing impact key='{key}', value='{value}': {e}")


        # Calculate averages
        for pos_data in position_impacts.values():
            if pos_data['count'] > 0 and pos_data['total'] is not None:
                pos_data['average'] = pos_data['total'] / pos_data['count']

        for len_data in length_impacts.values():
            if len_data['count'] > 0 and len_data['total'] is not None:
                len_data['average'] = len_data['total'] / len_data['count']

        # Calculate program type averages
        for impacts in program_type_impacts.values():
            if impacts['count'] > 0 and impacts['total_impact'] is not None:
                impacts['average'] = impacts['total_impact'] / impacts['count']
            else:
                impacts['average'] = None

        # Return all impact analyses
        return {
            'program_type_impacts': program_type_impacts,
            'position_impacts': position_impacts,
            'length_impacts': length_impacts,
            # Return the raw mean coefficients xarray DataArray too
            'channel_coefs_mean': channel_coefs_mean
        }

    def calculate_viewer_sensitivity(self):
        """Calculate viewer sensitivity metrics for different program types.

        This measures how sensitive viewers are to commercial breaks
        in different types of programs.

        Returns:
            dict: Sensitivity metrics by program type.
        """
        # Ensure model is fitted before analyzing
        if constants.POSTERIOR not in self.inference_data.groups():
            raise model.NotFittedModelError(
                "Calculating viewer sensitivity requires a fitted model."
            )

        # Get break impact analysis
        impact_analysis = self.analyze_break_impact() # This now calls the corrected method

        # Check if impact analysis returned expected structure
        if not impact_analysis or 'program_type_impacts' not in impact_analysis:
             logger.error("Impact analysis did not return the expected 'program_type_impacts'. Cannot calculate sensitivity.")
             return {} # Return empty dict

        # Calculate sensitivity
        sensitivity = {}

        for program_type, impacts in impact_analysis['program_type_impacts'].items():
            avg_impact = impacts.get('average') # Use .get for safety

            if avg_impact is not None:
                # Calculate variance of impacts for this program type
                impact_values = [
                    value for key, value in impacts.items()
                    if key not in ['total_impact', 'count', 'average'] and value is not None
                ]

                if len(impact_values) > 1:
                    variance = np.var(impact_values)
                    sensitivity_score = variance / (abs(avg_impact) + 1e-9) # Add epsilon

                    sensitivity[program_type] = {
                        'variance': float(variance),
                        'average_impact': float(avg_impact),
                        'sensitivity_score': float(sensitivity_score)
                    }
                elif len(impact_values) == 1:
                     sensitivity[program_type] = {
                        'variance': 0.0,
                        'average_impact': float(avg_impact),
                        'sensitivity_score': 0.0
                    }
                else:
                     sensitivity[program_type] = {
                        'variance': 0.0,
                        'average_impact': float(avg_impact),
                        'sensitivity_score': 0.0
                     }

        # Store for later use
        self.viewer_sensitivity = sensitivity

        return sensitivity

    def predict_viewer_impact(self, break_schedule):
        """Predict the impact of a break schedule on viewership.

        Args:
            break_schedule: DataFrame containing break schedule with columns:
                program_type, position, break_length, num_breaks
                Optionally: base_rate, viewing_points

        Returns:
            DataFrame: The input schedule with added impact predictions.
        """
        if not isinstance(break_schedule, pd.DataFrame):
             raise ValueError("break_schedule must be a pandas DataFrame")

        # Ensure model has coefficients (implies fitted)
        try:
            impact_analysis = self.analyze_break_impact() # Run analysis to get coefficients
            if 'channel_coefs_mean' not in impact_analysis:
                 raise RuntimeError("Could not retrieve 'channel_coefs_mean' from impact analysis.")
            channel_coefs_mean = impact_analysis['channel_coefs_mean']
            # Dynamically get the channel coordinate name
            media_channel_coord_name = next(iter(channel_coefs_mean.coords.keys()), None)
            if media_channel_coord_name is None or media_channel_coord_name not in channel_coefs_mean.dims:
                 # Fallback if dims/coords are unusual
                 media_channel_coord_name = 'media_channel'
                 if media_channel_coord_name not in channel_coefs_mean.coords:
                    raise RuntimeError(f"Cannot determine channel coordinate name from {list(channel_coefs_mean.coords.keys())}")

        except model.NotFittedModelError:
             raise model.NotFittedModelError("Cannot predict impact, model has not been fitted.")
        except Exception as e:
             raise RuntimeError(f"Cannot predict impact, failed to get model coefficients: {e}")


        # Make a copy of the input schedule
        predicted_schedule = break_schedule.copy()

        # Initialize prediction columns safely
        predicted_schedule['predicted_retention'] = pd.NA
        predicted_schedule['predicted_revenue'] = pd.NA


        # Process each row in the schedule
        for idx, row in predicted_schedule.iterrows():
            try:
                # Construct channel name (ensure required columns exist)
                prog_type = row['program_type']
                position = str(row['position']).lower() # Use lowercase consistently
                break_len_val = row['break_length']
                num_breaks_val = row['num_breaks']

                # Check for NaN/None in required fields
                if pd.isna(prog_type) or pd.isna(position) or pd.isna(break_len_val) or pd.isna(num_breaks_val):
                    logger.warning(f"Skipping prediction for row {idx} due to missing required data (program_type, position, break_length, or num_breaks).")
                    continue

                # Map break length to Short/Medium/Long
                if break_len_val < 60:
                    length_cat = 'short'
                elif break_len_val < 120:
                    length_cat = 'medium'
                else:
                    length_cat = 'long'

                # Construct channel name (match case used in training data/coeffs)
                channel_name = f"{prog_type}_{position.title()}_{length_cat.title()}"

                # Look up coefficient if channel exists in the coordinates
                if channel_name in channel_coefs_mean[media_channel_coord_name].values:
                    coef = float(channel_coefs_mean.sel({media_channel_coord_name: channel_name}).item())

                    baseline_retention = 0.85  # Assumed typical retention
                    impact = coef * num_breaks_val
                    retention = baseline_retention + impact
                    retention = max(min(retention, 1.0), 0.2) # Bounds [0.2, 1.0]

                    # Calculate revenue
                    base_rate = row.get('base_rate', 1000.0)
                    viewing_points = row.get('viewing_points', 1.0)
                    base_rate = float(base_rate) if pd.notna(base_rate) else 1000.0
                    viewing_points = float(viewing_points) if pd.notna(viewing_points) else 1.0
                    break_len_val_fl = float(break_len_val)

                    revenue = base_rate * viewing_points * retention * int(num_breaks_val) * (break_len_val_fl / 60.0)

                    predicted_schedule.loc[idx, 'predicted_retention'] = retention
                    predicted_schedule.loc[idx, 'predicted_revenue'] = revenue
                else:
                    logger.warning(f"Coefficient for channel '{channel_name}' not found for row {idx}. Using baseline.")
                    predicted_schedule.loc[idx, 'predicted_retention'] = 0.85
                    predicted_schedule.loc[idx, 'predicted_revenue'] = pd.NA

            except KeyError as e:
                 logger.warning(f"Missing expected column '{e}' in break_schedule row {idx}. Skipping prediction.")
            except Exception as e:
                 logger.warning(f"Error processing row {idx}: {e}. Skipping prediction.")


        # Convert columns to numeric types
        predicted_schedule['predicted_retention'] = pd.to_numeric(predicted_schedule['predicted_retention'], errors='coerce')
        predicted_schedule['predicted_revenue'] = pd.to_numeric(predicted_schedule['predicted_revenue'], errors='coerce')

        return predicted_schedule

    def predict_revenue_impact(self, break_schedule):
        """Predict the revenue impact of a break schedule."""
        predicted_schedule = self.predict_viewer_impact(break_schedule)
        total_revenue = predicted_schedule['predicted_revenue'].sum(skipna=True)
        return float(total_revenue) if pd.notna(total_revenue) else 0.0


# --- TVBreakOptimizer Class ---

class TVBreakOptimizer(optimizer.BudgetOptimizer):
    """Specialized optimizer for TV commercial breaks."""

    # Fixed: Renamed 'init' to '__init__', corrected super() call
    def __init__(self, mmm, unified_data=None):
        """Initialize the TV break optimizer."""
        # Call parent's initializer - assuming parent takes only 'mmm'
        super().__init__(mmm) # Correctly calls parent __init__
        self.mmm = mmm
        self.unified_data = unified_data
        self.viewer_sensitivity = None

        # If we have a TVBreakModel, try to get/calculate viewer sensitivity
        if isinstance(mmm, TVBreakModel):
            # Prefer pre-calculated sensitivity if available
            if getattr(mmm, 'viewer_sensitivity', None) is not None:
                 self.viewer_sensitivity = mmm.viewer_sensitivity
            else:
                 try:
                    # Calculate it only if the model has posterior samples
                    if constants.POSTERIOR in mmm.inference_data.groups():
                       self.viewer_sensitivity = mmm.calculate_viewer_sensitivity()
                    else:
                       logger.warning("Cannot calculate viewer sensitivity during optimizer init: Model not fitted.")
                       self.viewer_sensitivity = {}
                 except model.NotFittedModelError as e:
                     logger.warning(f"Cannot calculate viewer sensitivity during optimizer init: {e}")
                     self.viewer_sensitivity = {}
                 except Exception as e:
                    # Catch specific error related to constants if needed, or general Exception
                    logger.warning(f"Warning: Could not calculate viewer sensitivity during optimizer init: {e}")
                    self.viewer_sensitivity = {}
        else:
            logger.warning("Optimizer initialized with a model that is not a TVBreakModel. Viewer sensitivity features may not work.")
            self.viewer_sensitivity = {}

        # Crucial: Ensure the self.mmm attribute is indeed set by the parent call
        if not hasattr(self, 'mmm') or self.mmm is None:
             # This should ideally not happen if super().__init__(mmm) worked
             logger.error("Optimizer's parent class did not set the 'mmm' attribute correctly!")
             # Optionally raise an error here, or try setting it manually (less ideal)
             # self.mmm = mmm

    def optimize_allocation(self, **kwargs):
        # Ensure the model is available AND was set correctly during init
        if not hasattr(self, 'mmm') or self.mmm is None:
             raise RuntimeError("Optimizer not initialized properly, 'mmm' attribute missing or None.")

        # Ensure the model is fitted before optimization
        if constants.POSTERIOR not in self.mmm.inference_data.groups():
             raise model.NotFittedModelError("Optimization requires a fitted model.")

        min_viewer_retention = kwargs.pop('min_viewer_retention', 0.7)
        max_breaks_per_hour = kwargs.pop('max_breaks_per_hour', 3)
        program_schedule_input = kwargs.pop('program_schedule', None)

        # --- Standard Optimization ---
        try:
            standard_results_obj = super().optimize(**kwargs) # Returns OptimizationResults object

            # --- MODIFIED SECTION START ---
            # Access the 'optimized_data' attribute (which is a Dataset)
            if hasattr(standard_results_obj, 'optimized_data'):
                optimized_data_ds = standard_results_obj.optimized_data # This is the Dataset

                # Ensure it's a Dataset before trying to access variables
                if not isinstance(optimized_data_ds, xr.Dataset):
                    raise TypeError(f"Expected 'optimized_data' attribute to be an xarray Dataset, but got {type(optimized_data_ds)}")

                # Now, extract the SPEND DataArray from the Dataset
                spend_var_name = 'spend' # Common name for spend within the dataset
                if spend_var_name in optimized_data_ds:
                    optimized_spend_da = optimized_data_ds[spend_var_name] # Extract the DataArray
                else:
                    # Try another common name or list available vars for debugging
                    spend_var_name_alt = 'optimal_spend' # Less likely based on previous error
                    if spend_var_name_alt in optimized_data_ds:
                        optimized_spend_da = optimized_data_ds[spend_var_name_alt]
                        logger.warning(f"Using '{spend_var_name_alt}' variable from optimized_data Dataset.")
                    else:
                        available_vars = list(optimized_data_ds.data_vars.keys())
                        raise KeyError(f"Could not find spend variable ('{spend_var_name}' or '{spend_var_name_alt}') within the optimized_data Dataset. Available vars: {available_vars}")
            else:
                available_attrs = [attr for attr in dir(standard_results_obj) if not attr.startswith('_')]
                raise AttributeError(f"OptimizationResults object does not have 'optimized_data' attribute. Available attributes: {available_attrs}")

            # Ensure the extracted variable IS an xarray DataArray now
            if not isinstance(optimized_spend_da, xr.DataArray):
                 # This error should be much less likely now, but keep for safety
                 raise TypeError(f"Extracted spend variable ('{optimized_spend_da.name}') is not an xarray DataArray, but got {type(optimized_spend_da)}")
            # --- MODIFIED SECTION END ---

        except model.NotFittedModelError as e:
             logger.error(f"Optimization failed: {e}")
             raise e
        except (AttributeError, KeyError, TypeError) as e: # Catch specific expected errors
             logger.error(f"Optimization result structure unexpected: {e}")
             raise RuntimeError(f"Standard optimization failed: {e}")
        except Exception as e:
            logger.error(f"Error during standard optimization: {e}")
            raise RuntimeError(f"Standard optimization failed: {e}")

    # --- DEDENT THE FOLLOWING BLOCK ---
    # Ensure this block starts at the same indentation level as the 'try' above

        # --- TV-Specific Post-Processing ---
        if program_schedule_input is not None:
            program_schedule = program_schedule_input
        else:
            program_schedule = self._create_default_program_schedule()

        break_schedule = self._map_allocation_to_breaks(
            optimized_spend_da, # Pass the correctly extracted DataArray
            program_schedule,
            min_viewer_retention,
            max_breaks_per_hour
        )

        tv_results = {
            'break_schedule': break_schedule,
            'program_schedule': program_schedule,
            # Store the whole results object for context if needed
            'standard_optimization_results_object': standard_results_obj
        }
        return tv_results

    def _create_default_program_schedule(self):
        """Create a default program schedule DataFrame if none is provided."""
        # Default program types - try to get from model if possible
        program_types = []
        if hasattr(self.mmm, 'input_data') and hasattr(self.mmm.input_data, 'media_channel') and self.mmm.input_data.media_channel is not None:
             all_channels = self.mmm.input_data.media_channel
             # Extract unique program types from media channel names
             program_types = sorted(list(set(str(ch).split('_')[0] for ch in all_channels if '_' in str(ch))))

        # Ensure all necessary types, including those receiving spend, are present
        # Add types found in the logs ('News', 'Other', 'Promo') if not already parsed
        required_types = ['News', 'Other', 'Promo', 'Drama', 'Comedy', 'Reality', 'Documentary', 'Sports'] # Add all expected/possible types
        for req_type in required_types:
            if req_type not in program_types:
                program_types.append(req_type)
        program_types = sorted(list(set(program_types))) # Ensure unique and sorted

        if not program_types: # Absolute fallback
             program_types = ['News', 'Drama', 'Comedy', 'Reality', 'Documentary', 'Other', 'Promo', 'Sports']
             logger.warning(f"Could not determine program types reliably. Using comprehensive default list: {program_types}")
        else:
            logger.info(f"Using program types for default schedule: {program_types}")


        # Default time slots (hour of day)
        time_slots = list(range(8, 24))  # 8am to 11pm (exclusive of 24)

        # Create simplified schedule
        schedule_data = []
        # Adjust probabilities if needed, ensure all types have a chance to be selected
        n_types = len(program_types)
        weights = np.ones(n_types) / n_types # Start with uniform probability

        for hour in time_slots:
            # Simplified selection - uses uniform probability across all available types
            # You could refine this logic based on time of day if desired
            prog_type = np.random.choice(program_types, p=weights)

            prime_time = 18 <= hour < 23
            viewing_points = 3.0 if prime_time else (1.5 if hour >= 12 else 1.0)
            # Adjust duration logic slightly if needed for new types
            duration = 60 if prime_time or prog_type in ['Drama', 'News', 'Sports'] else 30
            if prog_type in ['Other', 'Promo']: # Assign default duration for these
                 duration = 30

            schedule_data.append({
                'hour': hour, 'program_type': prog_type, 'duration': duration,
                'prime_time': prime_time, 'viewing_points': viewing_points
            })
        return pd.DataFrame(schedule_data)


    def _map_allocation_to_breaks(self, optimized_spend_da, program_schedule,
                                 min_viewer_retention, max_breaks_per_hour):
        """Map optimized budget/spend allocation to commercial break schedule."""
        # <<<--- START DEBUGGING ADDITIONS --->>>
        logger.info(f"--- Mapping Allocation ---")
        logger.info(f"Min Viewer Retention Threshold: {min_viewer_retention}")
        logger.info(f"Max Breaks Per Hour: {max_breaks_per_hour}")
        logger.info(f"Optimized Spend DataArray:\n{optimized_spend_da}")
        # <<<--- END DEBUGGING ADDITIONS --->>>

        if not isinstance(optimized_spend_da, xr.DataArray):
             raise TypeError(f"optimized_spend_da must be an xarray DataArray, got {type(optimized_spend_da)}")

        # Identify the coordinate/dimension name for channels more robustly
        channel_coord_name = None
        # ... (rest of channel_coord_name finding logic remains the same) ...
        possible_names = [
            constants.MEDIA_CHANNEL if hasattr(constants, 'MEDIA_CHANNEL') else None,
            'media_channel', 'channel', 'media'
        ]
        for name in possible_names:
            if name and name in optimized_spend_da.coords: channel_coord_name = name; break
            elif name and name in optimized_spend_da.dims: channel_coord_name = name; break
        if channel_coord_name is None:
             available_coords = list(optimized_spend_da.coords.keys()); available_dims = list(optimized_spend_da.dims)
             raise KeyError(f"Could not find channel coord/dim. Tried: {possible_names}. Coords: {available_coords}. Dims: {available_dims}.")

        channels = optimized_spend_da[channel_coord_name].values

        # Handle potential geo dimension in spend values
        # ... (spend_values calculation logic remains the same) ...
        geo_dim_const = constants.GEO if hasattr(constants, 'GEO') else 'geo'
        actual_dims = list(optimized_spend_da.dims)
        if geo_dim_const in actual_dims:
             spend_values = optimized_spend_da.sum(dim=geo_dim_const).values
        elif len(actual_dims) > 1 and channel_coord_name in actual_dims:
             other_dims = [d for d in actual_dims if d != channel_coord_name]
             spend_values = optimized_spend_da.sum(dim=other_dims).values
        elif len(actual_dims) == 1 and actual_dims[0] == channel_coord_name:
             spend_values = optimized_spend_da.values
        else:
             spend_values = optimized_spend_da.values

        # <<<--- START DEBUGGING ADDITIONS --->>>
        # Check if any spend is non-zero
        if not np.any(spend_values > 0):
            logger.warning("Optimizer allocated zero spend to all channels.")
        # <<<--- END DEBUGGING ADDITIONS --->>>

        break_allocation = []
        # Use the identified channel_coord_name when iterating
        for channel, spend in zip(optimized_spend_da[channel_coord_name].values, spend_values):
            # Check if spend is positive before processing
            if spend > 0:
                 # Log the channel and its associated spend
                 logger.debug(f"Processing channel '{channel}' with spend = {spend:.4f}")

                 # Parse channel name to get program_type, position, break_type
                 parts = str(channel).split('_')
                 if len(parts) >= 3:
                    program_type = parts[0]
                    position = parts[1].lower()
                    break_type = parts[2].lower()

                    # Ensure the program schedule has the 'program_type' column needed for matching
                    if 'program_type' not in program_schedule.columns:
                        logger.warning(f"  -> `program_schedule` DataFrame missing 'program_type' column. Cannot map allocation for channel '{channel}'.")
                        continue # Skip to the next channel in the loop

                    # --- DEBUG LOG: Search for matching programs ---
                    logger.debug(f"  Searching for program_type='{program_type}' in schedule.")
                    matching_programs = program_schedule[program_schedule['program_type'] == program_type]

                    # --- DEBUG LOG: Check if matches were found ---
                    if matching_programs.empty:
                         logger.warning(f"  -> No matching programs found in schedule for type '{program_type}'. Skipping break allocation for channel '{channel}'.")
                         continue # Skip this channel if no programs match

                    # Proceed if matching programs were found
                    logger.debug(f"  -> Found {len(matching_programs)} matching program(s).")

                    # Calculate average characteristics from matching programs
                    required_cols = ['viewing_points', 'duration', 'prime_time']
                    has_all_cols = all(col in matching_programs.columns for col in required_cols)
                    if not has_all_cols:
                         logger.warning(f"    -> Matching programs for '{program_type}' missing required columns ({required_cols}). Using defaults for calculations.")

                    avg_viewing_points = matching_programs['viewing_points'].mean() if has_all_cols else 1.0
                    avg_duration = matching_programs['duration'].mean() if has_all_cols else 60.0 # Duration in minutes
                    prime_time_ratio = matching_programs['prime_time'].mean() if has_all_cols else 0.0

                    # Determine break length in seconds
                    break_length_sec = {'short': 45, 'medium': 90, 'long': 180}.get(break_type, 90) # Default medium

                    # Calculate number of breaks based on spend (assuming spend = total seconds)
                    total_break_time_sec = spend
                    num_breaks_calculated = total_break_time_sec / break_length_sec if break_length_sec > 0 else 0

                    # Apply constraint: max breaks per hour
                    if not has_all_cols or 'duration' not in matching_programs.columns:
                        total_hours_for_type = 1.0 # Default if duration info missing
                        logger.warning("    -> Missing duration column for max breaks constraint calculation. Using default hours.")
                    else:
                        total_duration_minutes = pd.to_numeric(matching_programs['duration'], errors='coerce').fillna(0).sum()
                        total_hours_for_type = total_duration_minutes / 60.0

                    max_total_breaks_allowed = total_hours_for_type * max_breaks_per_hour if total_hours_for_type > 0 else 0

                    # Determine final number of breaks after applying constraints
                    num_breaks = int(round(min(num_breaks_calculated, max_total_breaks_allowed)))
                    num_breaks = max(0, num_breaks) # Ensure non-negative

                    # --- DEBUG LOG: Log break calculation details ---
                    logger.debug(f"  -> Spend: {spend:.2f}, Calc'd Breaks: {num_breaks_calculated:.2f}, Max Allowed: {max_total_breaks_allowed:.2f}, Final Breaks: {num_breaks}")

                    # Only append if the final number of breaks is greater than zero
                    if num_breaks > 0:
                        # Calculate example base rate per viewing point
                        base_rate_per_vp = 1000 * (1 + prime_time_ratio * 0.5)

                        # --- DEBUG LOG: Confirm appending ---
                        logger.debug(f"    -> Appending {num_breaks} breaks to allocation for channel '{channel}'.")

                        # Append the dictionary with break details
                        break_allocation.append({
                            'program_type': program_type,
                            'position': position,
                            'break_type': break_type,
                            'break_length': break_length_sec,
                            'num_breaks': num_breaks,
                            'total_break_time': num_breaks * break_length_sec,
                            'viewing_points': avg_viewing_points,
                            'base_rate': base_rate_per_vp,
                            'program_duration': avg_duration, # In minutes
                            'prime_time_ratio': prime_time_ratio
                        })
                    else:
                         # --- DEBUG LOG: Log why breaks were not appended ---
                         logger.debug(f"    -> Final num_breaks is 0 for channel '{channel}'. Not appending.")
                 else:
                     # --- DEBUG LOG: Channel name parsing issue ---
                     logger.warning(f"  Could not parse channel name '{channel}' into 3+ parts. Skipping.")


        break_df = pd.DataFrame(break_allocation)

        # --- Predict impact and filter ---
        # <<<--- START DEBUGGING MODIFICATION --->>>
        # Temporarily use a very low retention threshold for debugging the mapping part
        # debug_min_viewer_retention = 0.0 # Set extremely low to bypass filtering for now
        # logger.info(f"Applying DEBUG retention threshold: {debug_min_viewer_retention} (Original: {min_viewer_retention})")
        effective_min_retention = min_viewer_retention # Use the debug value
        # <<<--- END DEBUGGING MODIFICATION --->>>

        if isinstance(self.mmm, TVBreakModel) and not break_df.empty:
            logger.debug(f"Predicting impact for {len(break_df)} potential break allocations...")
            try:
                break_df_with_preds = self.mmm.predict_viewer_impact(break_df)

                # <<<--- START DEBUGGING ADDITIONS --->>>
                if 'predicted_retention' in break_df_with_preds.columns:
                     logger.debug(f"Predicted Retentions (before filtering):\n{break_df_with_preds[['program_type', 'position', 'break_type', 'num_breaks', 'predicted_retention']]}")
                else:
                     logger.warning("predict_viewer_impact did not add 'predicted_retention' column.")
                # <<<--- END DEBUGGING ADDITIONS --->>>

                # Filter breaks that don't meet minimum retention, handle NAs
                if 'predicted_retention' in break_df_with_preds.columns:
                    retention_numeric = pd.to_numeric(break_df_with_preds['predicted_retention'], errors='coerce').fillna(0)
                    # USE the effective_min_retention for filtering during debug
                    break_df_filtered = break_df_with_preds[retention_numeric >= effective_min_retention].copy()
                    # <<<--- START DEBUGGING ADDITIONS --->>>
                    logger.info(f"Filtered breaks: {len(break_df_filtered)} remaining out of {len(break_df)} after retention filter.")
                    # <<<--- END DEBUGGING ADDITIONS --->>>
                    return break_df_filtered
                else:
                    logger.warning("Cannot filter by retention, column missing.")
                    return break_df_with_preds # Return unfiltered
            except model.NotFittedModelError as e:
                logger.error(f"Cannot predict impact during mapping: {e}")
                return break_df
            except Exception as e:
                 logger.error(f"Warning: Failed to predict/filter breaks during mapping: {e}")
                 return break_df
        else:
             if break_df.empty: logger.debug("No break allocations generated before prediction step.")
             else: logger.debug("Model is not TVBreakModel type, skipping prediction/filtering.")
             return break_df # Return if no model or empty allocation


    def optimize_weekly_schedule(self, program_schedule, **kwargs):
        """Optimize commercial breaks for a weekly program schedule."""
        if not isinstance(program_schedule, pd.DataFrame) or 'day' not in program_schedule.columns:
             raise ValueError("program_schedule must be a DataFrame with a 'day' column.")

        # Ensure model is fitted
        if constants.POSTERIOR not in self.mmm.inference_data.groups():
             raise model.NotFittedModelError("Weekly optimization requires a fitted model.")

        base_min_retention = kwargs.pop('min_viewer_retention', 0.75)
        base_max_breaks = kwargs.pop('max_breaks_per_hour', 3)
        days = program_schedule['day'].unique()
        daily_schedules = []

        for day in days:
            day_programs = program_schedule[program_schedule['day'] == day].copy()
            if day_programs.empty:
                logger.warning(f"No programs for day: {day}. Skipping.")
                continue

            day_kwargs = kwargs.copy()
            day_kwargs['min_viewer_retention'] = base_min_retention
            day_kwargs['max_breaks_per_hour'] = base_max_breaks
            if day in ['Saturday', 'Sunday']:
                day_kwargs['min_viewer_retention'] = min(0.8, base_min_retention + 0.05)
                day_kwargs['max_breaks_per_hour'] = max(2, base_max_breaks - 1)

            try:
                day_results = self.optimize_allocation(program_schedule=day_programs, **day_kwargs)
                day_break_schedule = day_results.get('break_schedule')
                if isinstance(day_break_schedule, pd.DataFrame) and not day_break_schedule.empty:
                    day_break_schedule['day'] = day
                    daily_schedules.append(day_break_schedule)
                elif day_break_schedule is not None:
                     logger.warning(f"No breaks allocated for {day}.")

            except model.NotFittedModelError as e: raise e # Propagate critical error
            except Exception as e: logger.error(f"Error optimizing schedule for {day}: {e}")

        if daily_schedules:
             return pd.concat(daily_schedules, ignore_index=True)
        else:
             logger.warning("No breaks allocated for any day.")
             expected_cols = [
                 'program_type', 'position', 'break_type', 'break_length', 'num_breaks',
                 'total_break_time', 'viewing_points', 'base_rate', 'program_duration',
                 'prime_time_ratio', 'predicted_retention', 'predicted_revenue', 'day'
             ]
             return pd.DataFrame(columns=expected_cols)


    def optimize_daily_schedule(self, program_schedule, ad_inventory=None, **kwargs):
        """Optimize commercial breaks for a daily schedule, optionally assigning ads."""
        if not isinstance(program_schedule, pd.DataFrame):
             raise ValueError("program_schedule must be a DataFrame.")

        if constants.POSTERIOR not in self.mmm.inference_data.groups():
             raise model.NotFittedModelError("Daily optimization requires a fitted model.")

        kwargs.setdefault('min_viewer_retention', 0.8)
        kwargs.setdefault('max_breaks_per_hour', 3)

        try:
            optimization_results = self.optimize_allocation(program_schedule=program_schedule, **kwargs)
            break_schedule = optimization_results.get('break_schedule')
            if not isinstance(break_schedule, pd.DataFrame):
                 logger.warning("optimize_allocation did not return a valid DataFrame.")
                 return pd.DataFrame()
        except model.NotFittedModelError as e: raise e # Propagate
        except Exception as e:
             logger.error(f"Error during daily break schedule optimization: {e}")
             raise RuntimeError(f"Daily optimization failed: {e}")

        if ad_inventory is not None and isinstance(ad_inventory, pd.DataFrame) \
           and not ad_inventory.empty and not break_schedule.empty:
             try:
                required_ad_cols = ['duration', 'remaining_count']
                if not all(col in ad_inventory.columns for col in required_ad_cols):
                     raise ValueError(f"ad_inventory missing required columns: {required_ad_cols}")
                return self._assign_ads_to_breaks(break_schedule, ad_inventory.copy())
             except Exception as e:
                logger.warning(f"Failed to assign ads to breaks: {e}. Returning schedule without assignments.")
                return break_schedule
        else:
            return break_schedule


    def _assign_ads_to_breaks(self, break_schedule, ad_inventory):
        """Assign specific ads to commercial breaks based on inventory."""
        schedule_with_ads = break_schedule.copy()
        schedule_with_ads['ad_assignments'] = [[] for _ in range(len(schedule_with_ads))]
        schedule_with_ads['filled_duration'] = 0.0
        schedule_with_ads['fill_ratio'] = 0.0

        # Prepare ad inventory
        ad_inventory_sorted = ad_inventory
        sort_cols, ascending_order = [], []
        if 'priority' in ad_inventory_sorted.columns:
             sort_cols.append('priority')
             ascending_order.append(False) # Higher priority first
             ad_inventory_sorted['priority'] = pd.to_numeric(ad_inventory_sorted['priority'], errors='coerce').fillna(0)

        ad_inventory_sorted['duration'] = pd.to_numeric(ad_inventory_sorted['duration'], errors='coerce').fillna(0)
        ad_inventory_sorted['remaining_count'] = pd.to_numeric(ad_inventory_sorted['remaining_count'], errors='coerce').fillna(0)

        if sort_cols:
            ad_inventory_sorted = ad_inventory_sorted.sort_values(sort_cols, ascending=ascending_order)

        # Iterate through breaks
        for idx, break_info in schedule_with_ads.iterrows():
            break_length_sec = break_info.get('break_length', 0)
            program_type = break_info.get('program_type')
            if break_length_sec <= 0: continue

            # Filter available, fitting ads
            potential_ads = ad_inventory_sorted[
                (ad_inventory_sorted['duration'] > 0) &
                (ad_inventory_sorted['duration'] <= break_length_sec) &
                (ad_inventory_sorted['remaining_count'] > 0)
            ].copy()

            # Sort by preference (type match, priority)
            current_sort_cols = list(sort_cols)
            current_ascending = list(ascending_order)
            if 'preferred_program_type' in potential_ads.columns and program_type is not None:
                potential_ads['type_match_score'] = potential_ads['preferred_program_type'].apply(
                    lambda pref: 1 if pd.notna(pref) and pref == program_type else 0)
                current_sort_cols.insert(0, 'type_match_score')
                current_ascending.insert(0, False)

            if current_sort_cols:
                 potential_ads = potential_ads.sort_values(current_sort_cols, ascending=current_ascending)

            # Assign greedily
            assigned_ads_list = []
            remaining_break_length = float(break_length_sec)
            filled_duration_for_break = 0.0

            for _, ad_details in potential_ads.iterrows():
                original_index = ad_details.name
                ad_duration = float(ad_details['duration'])

                if ad_inventory_sorted.loc[original_index, 'remaining_count'] <= 0: continue
                if ad_duration <= remaining_break_length:
                    assigned_ads_list.append({
                        'ad_id': ad_details.get('ad_id', original_index),
                        'duration': ad_duration,
                        'name': ad_details.get('name', f"Ad_{original_index}")
                    })
                    remaining_break_length -= ad_duration
                    filled_duration_for_break += ad_duration
                    ad_inventory_sorted.loc[original_index, 'remaining_count'] -= 1
                    if remaining_break_length < 1: break # Stop if almost full

            schedule_with_ads.at[idx, 'ad_assignments'] = assigned_ads_list
            schedule_with_ads.at[idx, 'filled_duration'] = filled_duration_for_break
            schedule_with_ads.at[idx, 'fill_ratio'] = filled_duration_for_break / break_length_sec

        return schedule_with_ads


# --- BreakSchedulePlanner Class ---

class BreakSchedulePlanner:
    """Tool for generating commercial break schedules at different time horizons."""

    def __init__(self, mmm, optimizer, unified_data=None):
        """Initialize the break schedule planner."""
        if not isinstance(optimizer, TVBreakOptimizer):
             raise TypeError("optimizer must be an instance of TVBreakOptimizer")
        if not isinstance(mmm, TVBreakModel) or not hasattr(mmm, 'predict_viewer_impact'):
             raise TypeError("mmm object must be a TVBreakModel with a 'predict_viewer_impact' method.")
        # Check if optimizer has the mmm attribute *after* its __init__ runs
        if not hasattr(optimizer, 'mmm') or optimizer.mmm is None:
             # This check ensures the optimizer was successfully initialized
             raise ValueError("The provided optimizer has not been properly initialized with a model.")

        self.mmm = mmm
        self.optimizer = optimizer
        self.unified_data = unified_data

    def generate_monthly_plan(self, **kwargs):
        """Generate a high-level monthly commercial break plan summary."""
        params = {'min_viewer_retention': 0.7, 'max_breaks_per_hour': 3}
        params.update(kwargs) # Allow overrides and budget

        try:
            representative_schedule = self.optimizer._create_default_program_schedule()
        except Exception as e:
             logger.error(f"Failed to create default program schedule for monthly plan: {e}")
             return self._format_plan({'error': 'Failed to create program schedule'}, 'monthly')

        if constants.POSTERIOR not in self.mmm.inference_data.groups():
             logger.error("Cannot generate monthly plan: Model is not fitted.")
             return self._format_plan({'error': 'Model not fitted', 'program_schedule': representative_schedule}, 'monthly')

        try:
            monthly_opt_results = self.optimizer.optimize_allocation(program_schedule=representative_schedule, **params)

            if not isinstance(monthly_opt_results, dict) or 'break_schedule' not in monthly_opt_results:
                 logger.warning("Monthly optimization failed or missing break_schedule.")
                 plan_data = {'error': 'Optimization failed or missing break_schedule',
                              'program_schedule': representative_schedule,
                              'standard_results': monthly_opt_results.get('standard_optimization_results_dataset')}
                 return self._format_plan(plan_data, 'monthly')

            break_schedule = monthly_opt_results.get('break_schedule', pd.DataFrame())
            if not isinstance(break_schedule, pd.DataFrame): break_schedule = pd.DataFrame()

            monthly_plan_data = {
                'break_schedule': break_schedule,
                'program_schedule': representative_schedule,
                 'total_revenue': None, 'avg_retention': None,
                 'standard_optimization_results': monthly_opt_results.get('standard_optimization_results_dataset')
            }
            if not break_schedule.empty:
                 if 'predicted_revenue' in break_schedule.columns:
                     monthly_plan_data['total_revenue'] = break_schedule['predicted_revenue'].sum(skipna=True)
                 if 'predicted_retention' in break_schedule.columns:
                     monthly_plan_data['avg_retention'] = break_schedule['predicted_retention'].mean(skipna=True)

        except model.NotFittedModelError as e:
            logger.error(f"Error generating monthly plan: {e}")
            monthly_plan_data = {'error': str(e), 'program_schedule': representative_schedule}
        except Exception as e:
             logger.error(f"Error generating monthly plan: {e}")
             monthly_plan_data = {'error': str(e), 'program_schedule': representative_schedule}

        return self._format_plan(monthly_plan_data, 'monthly')


    def generate_weekly_plan(self, program_schedule=None, **kwargs):
        """Generate a weekly commercial break plan."""
        params = {'min_viewer_retention': 0.75, 'max_breaks_per_hour': 3}
        params.update(kwargs)

        if program_schedule is None:
            try:
                program_schedule = self._create_default_weekly_schedule()
            except Exception as e:
                 logger.error(f"Failed to create default weekly program schedule: {e}")
                 return self._format_plan({'error': 'Failed to create program schedule'}, 'weekly')
        elif not isinstance(program_schedule, pd.DataFrame) or 'day' not in program_schedule.columns:
             raise ValueError("Provided program_schedule must be a DataFrame with a 'day' column.")

        if constants.POSTERIOR not in self.mmm.inference_data.groups():
             logger.error("Cannot generate weekly plan: Model is not fitted.")
             return self._format_plan({'error': 'Model not fitted', 'program_schedule': program_schedule}, 'weekly')

        try:
            weekly_break_schedule = self.optimizer.optimize_weekly_schedule(program_schedule=program_schedule, **params)
            if not isinstance(weekly_break_schedule, pd.DataFrame):
                 logger.warning("Weekly optimization did not return a DataFrame.")
                 weekly_break_schedule = pd.DataFrame()

            weekly_plan_data = {
                'break_schedule': weekly_break_schedule, 'program_schedule': program_schedule,
                'total_revenue': None, 'avg_retention': None,
                'revenue_by_day': {}, 'retention_by_day': {}
            }

            if not weekly_break_schedule.empty:
                 if 'predicted_revenue' in weekly_break_schedule.columns:
                    weekly_plan_data['total_revenue'] = weekly_break_schedule['predicted_revenue'].sum(skipna=True)
                    if 'day' in weekly_break_schedule.columns:
                        revenue_by_day = weekly_break_schedule.groupby('day')['predicted_revenue'].sum()
                        weekly_plan_data['revenue_by_day'] = revenue_by_day.to_dict()

                 if 'predicted_retention' in weekly_break_schedule.columns:
                    weekly_plan_data['avg_retention'] = weekly_break_schedule['predicted_retention'].mean(skipna=True)
                    if 'day' in weekly_break_schedule.columns:
                        retention_by_day = weekly_break_schedule.groupby('day')['predicted_retention'].mean()
                        weekly_plan_data['retention_by_day'] = retention_by_day.to_dict()

        except model.NotFittedModelError as e:
             logger.error(f"Error generating weekly plan: {e}")
             weekly_plan_data = {'error': str(e), 'program_schedule': program_schedule}
        except Exception as e:
             logger.error(f"Error generating weekly plan: {e}")
             weekly_plan_data = {'error': str(e), 'program_schedule': program_schedule}

        return self._format_plan(weekly_plan_data, 'weekly')


    def generate_daily_plan(self, program_schedule, ad_inventory=None, **kwargs):
        """Generate a daily commercial break plan."""
        if not isinstance(program_schedule, pd.DataFrame):
             raise ValueError("program_schedule must be a DataFrame.")

        params = {'min_viewer_retention': 0.8, 'max_breaks_per_hour': 3}
        params.update(kwargs)

        if constants.POSTERIOR not in self.mmm.inference_data.groups():
             logger.error("Cannot generate daily plan: Model is not fitted.")
             return self._format_plan({'error': 'Model not fitted', 'program_schedule': program_schedule, 'ad_inventory': ad_inventory}, 'daily')

        try:
            daily_break_schedule = self.optimizer.optimize_daily_schedule(program_schedule=program_schedule, ad_inventory=ad_inventory, **params)
            if not isinstance(daily_break_schedule, pd.DataFrame):
                 logger.warning("Daily optimization did not return a DataFrame.")
                 daily_break_schedule = pd.DataFrame()

            daily_plan_data = {
                'break_schedule': daily_break_schedule, 'program_schedule': program_schedule,
                'ad_inventory': ad_inventory, 'total_revenue': None, 'avg_retention': None,
                'revenue_by_type': {}, 'retention_by_type': {}, 'used_inventory': {}
            }

            if not daily_break_schedule.empty:
                 if 'predicted_revenue' in daily_break_schedule.columns:
                    daily_plan_data['total_revenue'] = daily_break_schedule['predicted_revenue'].sum(skipna=True)
                    if 'program_type' in daily_break_schedule.columns:
                       revenue_by_type = daily_break_schedule.groupby('program_type')['predicted_revenue'].sum(skipna=True)
                       daily_plan_data['revenue_by_type'] = revenue_by_type.to_dict()

                 if 'predicted_retention' in daily_break_schedule.columns:
                    daily_plan_data['avg_retention'] = daily_break_schedule['predicted_retention'].mean(skipna=True)
                    if 'program_type' in daily_break_schedule.columns:
                        retention_by_type = daily_break_schedule.groupby('program_type')['predicted_retention'].mean(skipna=True)
                        daily_plan_data['retention_by_type'] = retention_by_type.to_dict()

                 if 'ad_assignments' in daily_break_schedule.columns and ad_inventory is not None:
                    used_inventory_count = {}
                    for assignments in daily_break_schedule['ad_assignments']:
                        if isinstance(assignments, list):
                             for ad in assignments:
                                ad_id = ad.get('ad_id')
                                if ad_id is not None: used_inventory_count[ad_id] = used_inventory_count.get(ad_id, 0) + 1
                    daily_plan_data['used_inventory'] = used_inventory_count

        except model.NotFittedModelError as e:
             logger.error(f"Error generating daily plan: {e}")
             daily_plan_data = {'error': str(e), 'program_schedule': program_schedule, 'ad_inventory': ad_inventory}
        except Exception as e:
             logger.error(f"Error generating daily plan: {e}")
             daily_plan_data = {'error': str(e), 'program_schedule': program_schedule, 'ad_inventory': ad_inventory}

        return self._format_plan(daily_plan_data, 'daily')


    def _create_default_weekly_schedule(self):
        """Create a default weekly program schedule DataFrame with 'day' column."""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # Get program types from optimizer's model (ensure consistency)
        program_types = []
        if hasattr(self.optimizer, 'mmm') and hasattr(self.optimizer.mmm, 'input_data') \
           and hasattr(self.optimizer.mmm.input_data, 'media_channel') \
           and self.optimizer.mmm.input_data.media_channel is not None:
             all_channels = self.optimizer.mmm.input_data.media_channel
             program_types = sorted(list(set(str(ch).split('_')[0] for ch in all_channels if '_' in str(ch))))

        # Ensure all necessary types are present
        required_types = ['News', 'Other', 'Promo', 'Drama', 'Comedy', 'Reality', 'Documentary', 'Sports']
        for req_type in required_types:
            if req_type not in program_types:
                program_types.append(req_type)
        program_types = sorted(list(set(program_types)))

        if not program_types: # Absolute fallback
             program_types = ['News', 'Drama', 'Comedy', 'Reality', 'Documentary', 'Other', 'Promo', 'Sports']
             logger.warning(f"Could not determine program types reliably for weekly schedule. Using comprehensive default list: {program_types}")
        else:
             logger.info(f"Using program types for default weekly schedule: {program_types}")


        schedule_data = []
        for day in days:
            is_weekend = day in ['Saturday', 'Sunday']
            n_types = len(program_types)

            # Adjust weights if needed, e.g., less Promo/Other on weekends?
            weights = np.ones(n_types)
            if is_weekend:
                if 'Sports' in program_types: weights[program_types.index('Sports')] *= 1.5
                if 'Drama' in program_types: weights[program_types.index('Drama')] *= 1.2
                if 'News' in program_types: weights[program_types.index('News')] *= 0.5
                if 'Promo' in program_types: weights[program_types.index('Promo')] *= 0.3 # Less promo?
                if 'Other' in program_types: weights[program_types.index('Other')] *= 0.5 # Less other?
            else:
                if 'News' in program_types: weights[program_types.index('News')] *= 1.5
                if 'Drama' in program_types: weights[program_types.index('Drama')] *= 1.2
            weights = np.array(weights) / sum(weights) if sum(weights) > 0 else np.ones(n_types) / n_types


            for hour in range(6, 24):
                prime_time = 18 <= hour < 23
                viewing_points = 3.0 if prime_time else (1.5 if hour >= 12 else 1.0)
                if is_weekend and not prime_time: viewing_points *= 1.2

                prog_type = np.random.choice(program_types, p=weights)

                # Adjust duration logic
                duration = 60 if prime_time or prog_type in ['Drama', 'Sports', 'News'] else 30
                if prog_type in ['Other', 'Promo']: duration = 30 # Assign default

                prog_name = f"{prog_type} {day} {hour}:00"
                schedule_data.append({
                    'day': day, 'hour': hour, 'program_type': prog_type, 'program_name': prog_name,
                    'duration': duration, 'prime_time': prime_time, 'viewing_points': viewing_points,
                    'is_weekend': is_weekend
                })
        return pd.DataFrame(schedule_data)


    def _format_plan(self, plan_data, plan_type):
        """Format the break plan dictionary for output."""
        if 'error' in plan_data:
             formatted_plan = {'plan_type': plan_type, 'status': 'error', 'message': plan_data['error'],
                               'break_schedule': [], 'program_schedule': [], 'summary': {}}
             program_schedule_df = plan_data.get('program_schedule')
             if isinstance(program_schedule_df, pd.DataFrame):
                  formatted_plan['program_schedule'] = program_schedule_df.replace({np.nan: None}).to_dict(orient='records')
             return formatted_plan

        break_schedule_df = plan_data.get('break_schedule', pd.DataFrame())
        program_schedule_df = plan_data.get('program_schedule', pd.DataFrame())
        if not isinstance(break_schedule_df, pd.DataFrame): break_schedule_df = pd.DataFrame()
        if not isinstance(program_schedule_df, pd.DataFrame): program_schedule_df = pd.DataFrame()

        # Calculate totals safely
        total_breaks = 0
        total_break_time = 0.0
        if not break_schedule_df.empty:
            if 'num_breaks' in break_schedule_df.columns:
                 total_breaks = pd.to_numeric(break_schedule_df['num_breaks'], errors='coerce').fillna(0).sum()
            else: total_breaks = len(break_schedule_df)
            if 'total_break_time' in break_schedule_df.columns:
                 total_break_time = pd.to_numeric(break_schedule_df['total_break_time'], errors='coerce').fillna(0).sum()
            elif 'break_length' in break_schedule_df.columns: # Calculate if possible
                 lengths = pd.to_numeric(break_schedule_df['break_length'], errors='coerce').fillna(0)
                 counts = pd.to_numeric(break_schedule_df.get('num_breaks', 1), errors='coerce').fillna(1) # Assume 1 if missing
                 total_break_time = (lengths * counts).sum()

        total_revenue = plan_data.get('total_revenue')
        avg_retention = plan_data.get('avg_retention')

        formatted_plan = {
            'plan_type': plan_type, 'status': 'success',
            'break_schedule': break_schedule_df.replace({np.nan: None}).to_dict(orient='records'),
            'program_schedule': program_schedule_df.replace({np.nan: None}).to_dict(orient='records'),
            'summary': {
                'total_breaks': int(total_breaks),
                'total_break_time_seconds': float(total_break_time),
                'total_predicted_revenue': float(total_revenue) if pd.notna(total_revenue) else None,
                'average_predicted_retention': float(avg_retention) if pd.notna(avg_retention) else None
            }
        }

        # Add Plan-Type Specific Aggregations if data exists
        if not break_schedule_df.empty:
            try:
                 agg_cols_base = {
                     'num_breaks': ('num_breaks', 'sum') if 'num_breaks' in break_schedule_df.columns else ('break_length', 'size'),
                     'avg_break_length': ('break_length', 'mean') if 'break_length' in break_schedule_df.columns else None,
                     'avg_predicted_retention': ('predicted_retention', 'mean') if 'predicted_retention' in break_schedule_df.columns else None,
                     'total_predicted_revenue': ('predicted_revenue', 'sum') if 'predicted_revenue' in break_schedule_df.columns else None
                 }
                 agg_cols = {k: v for k, v in agg_cols_base.items() if v is not None}
                 rename_map = {'break_length': 'count'} if 'num_breaks' not in break_schedule_df.columns else {}

                 if agg_cols:
                    if plan_type == 'monthly':
                        if 'program_type' in break_schedule_df.columns:
                            by_program_type = break_schedule_df.groupby('program_type').agg(**agg_cols).rename(columns=rename_map).reset_index()
                            formatted_plan['summary_by_program_type'] = by_program_type.replace({np.nan: None}).to_dict(orient='records')
                        if 'position' in break_schedule_df.columns:
                            by_position = break_schedule_df.groupby('position').agg(**agg_cols).rename(columns=rename_map).reset_index()
                            formatted_plan['summary_by_position'] = by_position.replace({np.nan: None}).to_dict(orient='records')

                    elif plan_type == 'weekly':
                        if 'day' in break_schedule_df.columns:
                            by_day = break_schedule_df.groupby('day').agg(**agg_cols).rename(columns=rename_map).reset_index()
                            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            by_day['day'] = pd.Categorical(by_day['day'], categories=day_order, ordered=True)
                            by_day = by_day.sort_values('day')
                            formatted_plan['summary_by_day'] = by_day.replace({np.nan: None}).to_dict(orient='records')
                        formatted_plan['summary']['revenue_by_day'] = plan_data.get('revenue_by_day', {})
                        formatted_plan['summary']['retention_by_day'] = plan_data.get('retention_by_day', {})

                    elif plan_type == 'daily':
                        if 'hour' in program_schedule_df.columns and 'program_type' in break_schedule_df.columns and 'program_type' in program_schedule_df.columns:
                            try:
                                schedule_for_hourly = pd.merge(
                                     break_schedule_df,
                                     program_schedule_df[['program_type', 'hour']].drop_duplicates(subset=['program_type', 'hour']),
                                     on='program_type', how='left')
                                if 'hour' in schedule_for_hourly.columns and not schedule_for_hourly['hour'].isnull().all():
                                     schedule_for_hourly['hour'] = pd.to_numeric(schedule_for_hourly['hour'], errors='coerce').astype('Int64')
                                     valid_hourly_data = schedule_for_hourly.dropna(subset=['hour'])
                                     if not valid_hourly_data.empty:
                                         by_hour = valid_hourly_data.groupby('hour').agg(**agg_cols).rename(columns=rename_map).reset_index().sort_values('hour')
                                         formatted_plan['summary_by_hour'] = by_hour.replace({np.nan: None}).to_dict(orient='records')
                            except Exception as merge_agg_e: logger.error(f"Error during daily hourly aggregation: {merge_agg_e}")

                        formatted_plan['summary']['revenue_by_program_type'] = plan_data.get('revenue_by_type', {})
                        formatted_plan['summary']['retention_by_program_type'] = plan_data.get('retention_by_type', {})
                        formatted_plan['summary']['used_ad_inventory_count'] = plan_data.get('used_inventory', {})

                        # Ensure ad_assignments are serializable
                        for row_dict in formatted_plan['break_schedule']:
                             if 'ad_assignments' in row_dict:
                                 if not isinstance(row_dict['ad_assignments'], list):
                                     row_dict['ad_assignments'] = [] if pd.isna(row_dict['ad_assignments']) else list(row_dict['ad_assignments'])

            except Exception as e:
                 logger.warning(f"Error during {plan_type} plan aggregation: {e}")

        return formatted_plan


# --- Example Usage Placeholder ---
# (Keep commented out - requires actual data loading and model training steps)
#
# Example Workflow:
#
# 1. Load and transform data using TVBreakDataTransformer
#    transformer = TVBreakDataTransformer(
#        dayparts_path='path/to/dayparts.xlsx',
#        programmes_path='path/to/programmes.xlsx',
#        spots_path='path/to/spots.xlsx'
#    )
#    data_result = transformer.transform_data()
#
#    # Check if data transformation was successful
#    if data_result and data_result.get('meridian_data'):
#        meridian_data = data_result['meridian_data']
#        unified_data = data_result['aggregated_data'] # Optional for context
#    else:
#        print("ERROR: Data transformation failed.")
#        exit()
#
# 2. Create and train the TVBreakModel
#    model_spec = TVBreakModelSpec() # Uses the updated class
#    mmm = TVBreakModel(meridian_data, model_spec)
#
#    print("Sampling prior...")
#    mmm.sample_prior(500) # Adjust n_draws as needed
#
#    print("Sampling posterior...")
#    try:
#        mmm.sample_posterior(
#            n_chains=4, # Adjust as per resources
#            n_adapt=500,
#            n_burnin=500,
#            n_keep=1000
#        )
#    except Exception as e:
#        print(f"ERROR during posterior sampling: {e}")
#        # Decide how to proceed - exit or try analysis with limited data?
#        exit()
#
#    # Save the fitted model (optional but recommended)
#    try:
#        from meridian.model.model import save_mmm
#        model_save_path = 'output/fitted_tv_break_model.pkl'
#        save_mmm(mmm, model_save_path)
#        print(f"Fitted model saved to {model_save_path}")
#    except Exception as e:
#        print(f"Warning: Failed to save model: {e}")
#
# 3. Initialize Optimizer and Planner
#    # If loading a saved model:
#    # from meridian.model.model import load_mmm
#    # mmm_fitted = load_mmm('output/fitted_tv_break_model.pkl')
#    mmm_fitted = mmm # Use the model trained in this session
#
#    break_optimizer = TVBreakOptimizer(mmm_fitted, unified_data)
#    break_planner = BreakSchedulePlanner(mmm_fitted, break_optimizer, unified_data)
#    print("Optimizer and Planner initialized.")
#
# 4. Generate Plans
#    # NOTE: The following plan generation steps might produce empty results
#    # (zero breaks, zero revenue) if the default program schedules generated
#    # internally do not contain program types ('News', 'Other', 'Promo', etc.)
#    # that match the channels the optimizer allocated spend to.
#    # Ensure _create_default_program_schedule and _create_default_weekly_schedule
#    # methods generate schedules that align with your modeled media channels,
#    # OR provide custom program_schedule DataFrames to the generate methods.
#
#    # Monthly Plan (uses default representative schedule)
#    try:
#        print("\nGenerating Monthly Plan...")
#        # Pass optimizer arguments like budget via kwargs
#        monthly_plan = break_planner.generate_monthly_plan(budget=100000)
#        print("Monthly Plan Summary:")
#        print(f"  Status: {monthly_plan.get('status')}")
#        if monthly_plan.get('status') == 'success':
#             summary = monthly_plan.get('summary', {})
#             print(f"  Total Breaks Allocated: {summary.get('total_breaks', 0)}")
#             print(f"  Avg Predicted Retention: {summary.get('average_predicted_retention', 'N/A')}")
#             print(f"  Total Predicted Revenue: {summary.get('total_predicted_revenue', 'N/A')}")
#             # You can add more details from monthly_plan['summary'] if needed
#        else:
#            print(f"  Error: {monthly_plan.get('message')}")
#        # Optional: Save the full plan details
#        # import json
#        # with open('output/monthly_plan.json', 'w') as f: json.dump(monthly_plan, f, indent=2)
#    except Exception as e:
#        print(f"ERROR generating monthly plan: {e}")
#
#    # Weekly Plan (uses default weekly schedule if none provided)
#    try:
#        print("\nGenerating Weekly Plan...")
#        # weekly_prog_schedule_df = pd.read_csv('path/to/your/weekly_schedule.csv') # Example loading custom schedule
#        # If weekly_prog_schedule_df is not provided, it uses _create_default_weekly_schedule
#        weekly_plan = break_planner.generate_weekly_plan(budget=20000) # Example budget
#        print("Weekly Plan Summary:")
#        print(f"  Status: {weekly_plan.get('status')}")
#        if weekly_plan.get('status') == 'success':
#            summary = weekly_plan.get('summary', {})
#            print(f"  Total Breaks Allocated: {summary.get('total_breaks', 0)}")
#            print(f"  Avg Predicted Retention: {summary.get('average_predicted_retention', 'N/A')}")
#            print(f"  Total Predicted Revenue: {summary.get('total_predicted_revenue', 'N/A')}")
#            print(f"  Predicted Revenue by Day: {summary.get('revenue_by_day', {})}")
#        else:
#            print(f"  Error: {weekly_plan.get('message')}")
#        # Optional: Save the full plan details
#        # with open('output/weekly_plan.json', 'w') as f: json.dump(weekly_plan, f, indent=2)
#        # The main script already saves this plan as a pickle and the schedule as CSV
#    except Exception as e:
#        print(f"ERROR generating weekly plan: {e}")
#
#    # Daily Plan (requires a specific daily schedule)
#    try:
#        print("\nGenerating Daily Plan (Example for Monday)...")
#        # Create or load a specific daily schedule DataFrame
#        default_weekly_sched = break_planner._create_default_weekly_schedule() # Get the default structure
#        daily_prog_schedule_df = default_weekly_sched[default_weekly_sched['day'] == 'Monday'].copy()
#
#        # Create or load ad inventory DataFrame (optional)
#        ad_inventory_df = pd.DataFrame({ # Example inventory
#            'ad_id': ['AD001', 'AD002', 'AD003', 'AD004'],
#            'name': ['BrandX 30s', 'BrandY 15s', 'BrandX 30s Prime', 'BrandZ 60s'],
#            'duration': [30, 15, 30, 60],
#            'remaining_count': [10, 20, 5, 8],
#            'priority': [1, 2, 3, 1], # Higher is better
#            'preferred_program_type': ['Drama', 'Comedy', 'Drama', None]
#        })
#
#        daily_plan = break_planner.generate_daily_plan(
#            program_schedule=daily_prog_schedule_df,
#            ad_inventory=ad_inventory_df, # Pass inventory here
#            budget=3000 # Example budget
#        )
#        print("Daily Plan Summary (Monday):")
#        print(f"  Status: {daily_plan.get('status')}")
#        if daily_plan.get('status') == 'success':
#             summary = daily_plan.get('summary', {})
#             print(f"  Total Breaks Allocated: {summary.get('total_breaks', 0)}")
#             print(f"  Avg Predicted Retention: {summary.get('average_predicted_retention', 'N/A')}")
#             print(f"  Total Predicted Revenue: {summary.get('total_predicted_revenue', 'N/A')}")
#             print(f"  Used Ad Inventory Count: {summary.get('used_ad_inventory_count', {})}")
#             # print("  Break Schedule:", daily_plan['break_schedule']) # Uncomment to see full schedule details
#        else:
#             print(f"  Error: {daily_plan.get('message')}")
#        # Optional: Save the full plan details
#        # with open('output/daily_plan_monday.json', 'w') as f: json.dump(daily_plan, f, indent=2)
#    except Exception as e:
#        print(f"ERROR generating daily plan: {e}")
#
# print("\nPlaceholder execution finished.")