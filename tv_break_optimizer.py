import os
import pandas as pd
import numpy as np
import argparse
import datetime as dt
import tensorflow as tf
import logging

# Import custom modules
from tv_break_data_transformer import TVBreakDataTransformer
from tv_break_model import TVBreakModel, TVBreakModelSpec, TVBreakOptimizer, BreakSchedulePlanner

# Configure logging
logging.basicConfig(
    # CHANGE level=logging.INFO to level=logging.DEBUG
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tv_break_optimizer')

def run_tv_break_optimization(
    dayparts_path, 
    programmes_path, 
    spots_path, 
    output_dir,
    planning_horizon='weekly',
    min_viewer_retention=0.75,
    max_breaks_per_hour=3,
    use_optimal_frequency=True,
    n_mcmc_chains=7,
    n_mcmc_adapt=500,
    n_mcmc_burnin=500,
    n_mcmc_keep=1000,
    seed=42
):
    """
    Run end-to-end TV commercial break optimization.
    
    Args:
        dayparts_path: Path to dayparts Excel file
        programmes_path: Path to programmes Excel file
        spots_path: Path to spots Excel file
        output_dir: Directory to save outputs
        planning_horizon: 'monthly', 'weekly', or 'daily'
        min_viewer_retention: Minimum viewer retention threshold
        max_breaks_per_hour: Maximum number of breaks per hour
        use_optimal_frequency: Whether to use optimal frequency in modeling
        n_mcmc_chains: Number of MCMC chains for posterior sampling
        n_mcmc_adapt: Number of adaptation steps for MCMC
        n_mcmc_burnin: Number of burn-in steps for MCMC
        n_mcmc_keep: Number of posterior samples to keep
        seed: Random seed for reproducibility
    
    Returns:
        Dict containing optimization results and paths to output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting TV commercial break optimization")
    start_time = dt.datetime.now()
    
    # Step 1: Transform data
    logger.info("Step 1: Transforming data")
    transformer = TVBreakDataTransformer(dayparts_path, programmes_path, spots_path)
    data_result = transformer.transform_data()
    
    if data_result is None or data_result.get('meridian_data') is None:
        logger.error("Failed to transform data")
        return {"status": "error", "message": "Data transformation failed"}
    
    meridian_data = data_result['meridian_data']
    unified_data = data_result['aggregated_data']
    raw_data = data_result['raw_data']
    
    # Save transformed data
    pd.to_pickle(data_result, os.path.join(output_dir, 'transformed_data.pkl'))
    logger.info(f"Saved transformed data to {os.path.join(output_dir, 'transformed_data.pkl')}")
    
    # Step 2: Create and train model
    logger.info("Step 2: Creating and training model")
    model_spec = TVBreakModelSpec()
    mmm = TVBreakModel(meridian_data, model_spec)
    
    # Sample from prior distribution
    logger.info("Sampling from prior distribution")
    mmm.sample_prior(500, seed=seed)
    
    # Sample from posterior distribution
    logger.info("Sampling from posterior distribution")
    try:
        mmm.sample_posterior(
            n_chains=n_mcmc_chains,
            n_adapt=n_mcmc_adapt,
            n_burnin=n_mcmc_burnin,
            n_keep=n_mcmc_keep,
            seed=seed
        )
    except Exception as e:
        logger.error(f"Error sampling from posterior: {e}")
        logger.info("Attempting to continue with limited posterior samples")
    
    # Save the trained model
    model_path = os.path.join(output_dir, 'trained_model.pkl')
    try:
        from meridian.model.model import save_mmm
        save_mmm(mmm, model_path)
        logger.info(f"Saved trained model to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
    
    # Step 3: Analyze commercial break impact
    logger.info("Step 3: Analyzing commercial break impact")
    try:
        impact_analysis = mmm.analyze_break_impact()
        pd.to_pickle(impact_analysis, os.path.join(output_dir, 'impact_analysis.pkl'))
        logger.info(f"Saved impact analysis to {os.path.join(output_dir, 'impact_analysis.pkl')}")
        
        # Calculate viewer sensitivity
        sensitivity = mmm.calculate_viewer_sensitivity()
        pd.to_pickle(sensitivity, os.path.join(output_dir, 'viewer_sensitivity.pkl'))
        logger.info(f"Saved viewer sensitivity metrics to {os.path.join(output_dir, 'viewer_sensitivity.pkl')}")
    except Exception as e:
        logger.error(f"Error in break impact analysis: {e}")
        impact_analysis = None
        sensitivity = None
    
    # Step 4: Optimize commercial breaks
    logger.info("Step 4: Optimizing commercial breaks")
    break_optimizer = TVBreakOptimizer(mmm, unified_data)
    
    # Create planner for the selected planning horizon
    break_planner = BreakSchedulePlanner(mmm, break_optimizer, unified_data)
    
    # Generate plan based on planning horizon
    plan_result = None
    try:
        if planning_horizon == 'monthly':
            logger.info("Generating monthly plan")
            plan_result = break_planner.generate_monthly_plan(
                min_viewer_retention=min_viewer_retention,
                max_breaks_per_hour=max_breaks_per_hour,
                use_optimal_frequency=use_optimal_frequency
            )
        elif planning_horizon == 'weekly':
            logger.info("Generating weekly plan")
            plan_result = break_planner.generate_weekly_plan(
                min_viewer_retention=min_viewer_retention,
                max_breaks_per_hour=max_breaks_per_hour,
                use_optimal_frequency=use_optimal_frequency
            )
        elif planning_horizon == 'daily':
            logger.info("Generating daily plan")
            # For daily plan, we'd ideally use a specific program schedule
            # For this example, we'll use a default program schedule
            program_schedule = break_planner._create_default_weekly_schedule()
            # Filter to a single day
            if 'day' in program_schedule.columns:
                program_schedule = program_schedule[program_schedule['day'] == 'Monday']
            
            plan_result = break_planner.generate_daily_plan(
                program_schedule=program_schedule,
                min_viewer_retention=min_viewer_retention,
                max_breaks_per_hour=max_breaks_per_hour,
                use_optimal_frequency=use_optimal_frequency
            )
        else:
            logger.error(f"Unknown planning horizon: {planning_horizon}")
            return {"status": "error", "message": f"Unknown planning horizon: {planning_horizon}"}
        
        # Save plan result
        plan_path = os.path.join(output_dir, f'{planning_horizon}_plan.pkl')
        pd.to_pickle(plan_result, plan_path)
        logger.info(f"Saved {planning_horizon} plan to {plan_path}")
    except Exception as e:
        logger.error(f"Error generating {planning_horizon} plan: {e}")
        plan_result = None
    
    # Step 5: Generate reports and visualizations
    logger.info("Step 5: Generating reports and visualizations")
    report_paths = {}
    
    try:
        # Generate a simple summary CSV for the break schedule
        if plan_result and 'break_schedule' in plan_result:
            break_schedule = pd.DataFrame(plan_result['break_schedule'])
            break_schedule_path = os.path.join(output_dir, f'{planning_horizon}_break_schedule.csv')
            break_schedule.to_csv(break_schedule_path, index=False)
            report_paths['break_schedule'] = break_schedule_path
            logger.info(f"Saved break schedule to {break_schedule_path}")
        
        # Generate a summary CSV for the impact analysis
        if impact_analysis:
            # Save program type impacts
            if 'program_type_impacts' in impact_analysis:
                program_impacts = []
                for program_type, impacts in impact_analysis['program_type_impacts'].items():
                    impacts['program_type'] = program_type
                    program_impacts.append(impacts)
                
                if program_impacts:
                    program_impacts_df = pd.DataFrame(program_impacts)
                    program_impacts_path = os.path.join(output_dir, 'program_type_impacts.csv')
                    program_impacts_df.to_csv(program_impacts_path, index=False)
                    report_paths['program_type_impacts'] = program_impacts_path
                    logger.info(f"Saved program type impacts to {program_impacts_path}")
            
            # Save position impacts
            if 'position_impacts' in impact_analysis:
                position_impacts = []
                for position, impacts in impact_analysis['position_impacts'].items():
                    impacts['position'] = position
                    position_impacts.append(impacts)
                
                if position_impacts:
                    position_impacts_df = pd.DataFrame(position_impacts)
                    position_impacts_path = os.path.join(output_dir, 'position_impacts.csv')
                    position_impacts_df.to_csv(position_impacts_path, index=False)
                    report_paths['position_impacts'] = position_impacts_path
                    logger.info(f"Saved position impacts to {position_impacts_path}")
            
            # Save length impacts
            if 'length_impacts' in impact_analysis:
                length_impacts = []
                for length, impacts in impact_analysis['length_impacts'].items():
                    impacts['length'] = length
                    length_impacts.append(impacts)
                
                if length_impacts:
                    length_impacts_df = pd.DataFrame(length_impacts)
                    length_impacts_path = os.path.join(output_dir, 'length_impacts.csv')
                    length_impacts_df.to_csv(length_impacts_path, index=False)
                    report_paths['length_impacts'] = length_impacts_path
                    logger.info(f"Saved length impacts to {length_impacts_path}")
    except Exception as e:
        logger.error(f"Error generating reports: {e}")
    
    # Step 6: Return results
    end_time = dt.datetime.now()
    run_time = (end_time - start_time).total_seconds()
    
    logger.info(f"Completed TV commercial break optimization in {run_time:.1f} seconds")
    
    return {
        "status": "success",
        "planning_horizon": planning_horizon,
        "output_dir": output_dir,
        "report_paths": report_paths,
        "run_time": run_time,
        "plan_result": plan_result,
        "impact_analysis": impact_analysis,
        "viewer_sensitivity": sensitivity
    }

def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(description='TV Commercial Break Optimization')
    
    parser.add_argument('--dayparts', required=True, help='Path to dayparts Excel file')
    parser.add_argument('--programmes', required=True, help='Path to programmes Excel file')
    parser.add_argument('--spots', required=True, help='Path to spots Excel file')
    parser.add_argument('--output_dir', default='./output', help='Directory to save outputs')
    parser.add_argument('--planning_horizon', choices=['monthly', 'weekly', 'daily'], default='weekly',
                        help='Planning horizon for optimization')
    parser.add_argument('--min_viewer_retention', type=float, default=0.75,
                        help='Minimum viewer retention threshold')
    parser.add_argument('--max_breaks_per_hour', type=int, default=3,
                        help='Maximum number of breaks per hour')
    parser.add_argument('--use_optimal_frequency', action='store_true',
                        help='Use optimal frequency in modeling')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Run the optimization
    result = run_tv_break_optimization(
        dayparts_path=args.dayparts,
        programmes_path=args.programmes,
        spots_path=args.spots,
        output_dir=args.output_dir,
        planning_horizon=args.planning_horizon,
        min_viewer_retention=args.min_viewer_retention,
        max_breaks_per_hour=args.max_breaks_per_hour,
        use_optimal_frequency=args.use_optimal_frequency,
        seed=args.seed
    )
    
    # Print results summary
    if result['status'] == 'success':
        print(f"\nTV Commercial Break Optimization Completed Successfully")
        print(f"Planning Horizon: {result['planning_horizon']}")
        print(f"Output Directory: {result['output_dir']}")
        print(f"Run Time: {result['run_time']:.1f} seconds")
        print("\nOutput Files:")
        for report_name, report_path in result['report_paths'].items():
            print(f"  - {report_name}: {report_path}")
    else:
        print(f"\nError: {result['message']}")

if __name__ == "__main__":
    main()
