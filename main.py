import numpy as np
import os
import sys
import argparse
from datetime import datetime

# Import our modules
from chaotic_systems import rossler_system, simulate_system
from embedding import (time_delay_embedding, find_optimal_delay, 
                      estimate_correlation_dimension, mutual_information)
from visualization import ResultsManager

def parse_arguments():
    parser = argparse.ArgumentParser(description='Chaotic System Simulation and Analysis')
    
    # Simulation parameters
    parser.add_argument('--system', type=str, default='rossler', 
                       choices=['rossler'], help='Chaotic system to simulate')
    parser.add_argument('--a', type=float, default=0.2, 
                       help='Parameter a for Rossler system')
    parser.add_argument('--b', type=float, default=0.2, 
                       help='Parameter b for Rossler system')
    parser.add_argument('--c', type=float, default=5.7, 
                       help='Parameter c for Rossler system')
    parser.add_argument('--initial_state', type=str, default='1.0,1.0,1.0', 
                       help='Initial state (comma-separated)')
    parser.add_argument('--t_start', type=float, default=0.0, 
                       help='Start time for simulation')
    parser.add_argument('--t_end', type=float, default=500.0, 
                       help='End time for simulation')
    parser.add_argument('--dt', type=float, default=0.01, 
                       help='Time step for output')
    parser.add_argument('--discard_transient', type=float, default=100.0, 
                       help='Time to discard as transient')
    
    # Embedding parameters
    parser.add_argument('--embedding_var', type=int, default=0, 
                       help='Variable index to use for embedding (0=x, 1=y, 2=z)')
    parser.add_argument('--embedding_dim', type=int, default=3, 
                       help='Embedding dimension')
    parser.add_argument('--delay', type=int, default=None, 
                       help='Time delay (in indices). If None, calculated automatically')
    parser.add_argument('--max_delay_check', type=int, default=100, 
                       help='Maximum delay to check when finding optimal delay')
    
    # Dimension estimation parameters
    parser.add_argument('--num_scales', type=int, default=20, 
                       help='Number of scales to use for dimension estimation')
    
    # Output parameters
    parser.add_argument('--results_dir', type=str, default='results', 
                       help='Directory to save results')
    
    return parser.parse_args()

def main():
    """Main function to run the simulation and analysis."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Convert initial state from string to list of floats
    initial_state = [float(x) for x in args.initial_state.split(',')]
    
    # Initialize the results manager
    results_mgr = ResultsManager(args.results_dir)
    
    # Save parameters
    params = vars(args)
    params['initial_state'] = initial_state
    params['run_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results_mgr.save_parameters(params)
    
    # Phase 1: Simulation
    print("Phase 1: Simulating chaotic system...")
    
    # Set up system parameters based on the chosen system
    if args.system == 'rossler':
        system_func = rossler_system
        params = {'a': args.a, 'b': args.b, 'c': args.c}
        var_names = ['x', 'y', 'z']
    else:
        raise ValueError(f"Unknown system: {args.system}")
    
    # Run the simulation
    sim_results = simulate_system(
        system_func, 
        initial_state, 
        [args.t_start, args.t_end], 
        args.dt, 
        params,
        args.discard_transient
    )
    
    # Extract data
    time = sim_results['time']
    states = sim_results['states']
    
    # Plot the original time series
    for i, var_name in enumerate(var_names):
        results_mgr.plot_time_series(time, states[:, i], var_name)
    
    # Plot the original attractor (phase space)
    results_mgr.plot_phase_space(states, var_names=var_names, plot_3d=True)
    
    # 2D projections of original attractor
    for i in range(len(var_names)):
        for j in range(i+1, len(var_names)):
            results_mgr.plot_phase_space(
                states, 
                var_indices=(i, j), 
                var_names=(var_names[i], var_names[j]), 
                plot_3d=False
            )
    
    # Phase 2: Select the variable for embedding and perform embedding
    print("Phase 2: Performing attractor reconstruction...")
    
    # Extract the variable to use for embedding
    embedding_var_idx = args.embedding_var
    embedding_var_name = var_names[embedding_var_idx]
    time_series = states[:, embedding_var_idx]
    
    # Determine the optimal delay if not specified
    if args.delay is None:
        print("Calculating optimal delay using mutual information...")
        delays = np.arange(1, args.max_delay_check + 1)
        mi_values = np.zeros(args.max_delay_check)
        
        for d in range(1, args.max_delay_check + 1):
            x = time_series[:-d] if d > 0 else time_series
            y = time_series[d:] if d > 0 else time_series
            mi_values[d-1] = mutual_information(x, y)
        
        # Find the first minimum
        optimal_delay = find_optimal_delay(time_series, args.max_delay_check)
        print(f"Optimal delay: {optimal_delay}")
        
        # Plot mutual information
        results_mgr.plot_mutual_information(delays, mi_values, optimal_delay)
    else:
        optimal_delay = args.delay
        print(f"Using specified delay: {optimal_delay}")
    
    # Perform time-delay embedding
    embedded_data = time_delay_embedding(time_series, args.embedding_dim, optimal_delay)
    print(f"Embedded data shape: {embedded_data.shape}")
    
    # Plot the reconstructed attractor
    results_mgr.plot_embedded_attractor(
        embedded_data, 
        optimal_delay, 
        args.embedding_dim, 
        embedding_var_name
    )
    
    # Phase 3: Estimate correlation dimension
    print("Phase 3: Estimating correlation dimension...")
    
    # Perform dimension estimation
    dim_results = estimate_correlation_dimension(
        embedded_data, 
        num_scales=args.num_scales
    )
    
    # Plot dimension estimation results
    results_mgr.plot_correlation_dimension(dim_results, args.embedding_dim)
    
    # Report the estimated dimension
    dimension = dim_results['dimension']
    print(f"Estimated correlation dimension: {dimension:.3f}")
    
    # Generate conclusions
    conclusions = (
        f"The correlation dimension of the reconstructed attractor is estimated to be "
        f"D = {dimension:.3f}, using an embedding dimension of m = {args.embedding_dim} and "
        f"a time delay of τ = {optimal_delay}. "
        f"For the {args.system.capitalize()} system with parameters a = {args.a}, b = {args.b}, "
        f"c = {args.c}, this is " + 
        ("close to the expected theoretical value of approximately 2.0." 
         if 1.8 <= dimension <= 2.2 else 
         "different from the expected theoretical value of approximately 2.0, "
         "which may indicate issues with the reconstruction or estimation procedure.")
    )
    
    # Finalize the results
    results_mgr.finalize(conclusions)

if __name__ == "__main__":
    main() 