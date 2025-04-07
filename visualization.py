import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import datetime
import json
import markdown

class ResultsManager:
    """
    Manages results from chaotic system simulations and analyses.
    Handles plot generation, saving, and documentation.
    """
    
    def __init__(self, base_dir="results"):
        """
        Initialize the results manager.
        
        Parameters:
            base_dir (str): Base directory for saving results
        """
        self.base_dir = base_dir
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, self.run_id)
        self.plots_dir = os.path.join(self.run_dir, "plots")
        self.dashboard_file = os.path.join(self.run_dir, "dashboard.md")
        self.plots_info = []
        
        # Create directories if they don't exist
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def save_plot(self, plot_func, filename, title, description):
        """
        Save a plot with metadata.
        
        Parameters:
            plot_func (callable): Function that creates the plot
            filename (str): Filename for the plot (without extension)
            title (str): Plot title
            description (str): Description of the plot
        
        Returns:
            str: Path to the saved plot
        """
        # Create a new figure
        plt.figure(figsize=(10, 6))
        
        # Call the plotting function
        plot_func()
        
        # Add title
        plt.title(title)
        
        # Save the plot
        plot_path = os.path.join(self.plots_dir, f"{filename}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual markdown file for this plot
        md_path = os.path.join(self.plots_dir, f"{filename}.md")
        with open(md_path, 'w') as f:
            f.write(f"# {title}\n\n")
            f.write(f"{description}\n\n")
            f.write(f"![{title}](./{filename}.png)\n")
        
        # Add to plots info
        self.plots_info.append({
            'filename': filename,
            'title': title,
            'description': description,
            'path': plot_path
        })
        
        # Update dashboard
        self._update_dashboard()
        
        return plot_path
    
    def plot_time_series(self, time, values, variable_name="x", discard_points=0):
        """
        Plot a time series and save the result.
        
        Parameters:
            time (ndarray): Time points
            values (ndarray): Variable values
            variable_name (str): Name of the variable
            discard_points (int): Number of initial points to discard
            
        Returns:
            str: Path to the saved plot
        """
        def plot_func():
            plt.plot(time[discard_points:], values[discard_points:])
            plt.xlabel('Time')
            plt.ylabel(f'${variable_name}(t)$')
            plt.grid(True, alpha=0.3)
        
        title = f"Time Series of {variable_name}(t)"
        description = f"Time series of the {variable_name} variable from the simulation, "
        if discard_points > 0:
            description += f"with the first {discard_points} points discarded as transients."
        else:
            description += "showing the full trajectory including initial transients."
            
        return self.save_plot(
            plot_func, 
            f"time_series_{variable_name}", 
            title, 
            description
        )
    
    def plot_phase_space(self, states, var_indices=(0, 1, 2), var_names=('x', 'y', 'z'), 
                        discard_points=0, plot_3d=True):
        """
        Plot the phase space trajectory.
        
        Parameters:
            states (ndarray): State vectors
            var_indices (tuple): Indices of variables to plot
            var_names (tuple): Names of the variables
            discard_points (int): Number of initial points to discard
            plot_3d (bool): Whether to plot in 3D
            
        Returns:
            str: Path to the saved plot
        """
        states = states[discard_points:]
        
        if plot_3d and len(var_indices) == 3:
            def plot_func():
                ax = plt.figure().add_subplot(111, projection='3d')
                ax.plot(states[:, var_indices[0]], 
                        states[:, var_indices[1]], 
                        states[:, var_indices[2]], 
                        lw=0.5)
                ax.set_xlabel(f'${var_names[0]}$')
                ax.set_ylabel(f'${var_names[1]}$')
                ax.set_zlabel(f'${var_names[2]}$')
                
            title = f"3D Phase Space: {var_names[0]}-{var_names[1]}-{var_names[2]}"
            description = f"3D phase space plot showing the attractor in the {var_names[0]}-{var_names[1]}-{var_names[2]} space."
                
        else:
            # 2D projection
            def plot_func():
                plt.plot(states[:, var_indices[0]], states[:, var_indices[1]], lw=0.5)
                plt.xlabel(f'${var_names[0]}$')
                plt.ylabel(f'${var_names[1]}$')
                plt.grid(True, alpha=0.3)
                
            title = f"Phase Space Projection: {var_names[0]}-{var_names[1]}"
            description = f"2D projection of the phase space onto the {var_names[0]}-{var_names[1]} plane."
        
        return self.save_plot(
            plot_func, 
            f"phase_space_{'_'.join(var_names)}", 
            title, 
            description
        )
    
    def plot_embedded_attractor(self, embedded_data, tau, embedding_dim, 
                               original_var='x'):
        """
        Plot the reconstructed attractor from embedded data.
        
        Parameters:
            embedded_data (ndarray): Embedded data vectors
            tau (int): Time delay used
            embedding_dim (int): Embedding dimension
            original_var (str): Name of the original variable
            
        Returns:
            str: Path to the saved plot
        """
        if embedding_dim == 3:
            def plot_func():
                ax = plt.figure().add_subplot(111, projection='3d')
                ax.plot(embedded_data[:, 0], 
                        embedded_data[:, 1], 
                        embedded_data[:, 2], 
                        lw=0.5)
                ax.set_xlabel(f'${original_var}(t)$')
                ax.set_ylabel(f'${original_var}(t+{tau})$')
                ax.set_zlabel(f'${original_var}(t+{2*tau})$')
                
            title = f"Reconstructed Attractor (τ={tau}, m={embedding_dim})"
            description = f"3D reconstruction of the attractor using time-delay embedding with τ={tau} and embedding dimension m={embedding_dim}."
                
        elif embedding_dim == 2:
            def plot_func():
                plt.plot(embedded_data[:, 0], embedded_data[:, 1], lw=0.5)
                plt.xlabel(f'${original_var}(t)$')
                plt.ylabel(f'${original_var}(t+{tau})$')
                plt.grid(True, alpha=0.3)
                
            title = f"Reconstructed Attractor (τ={tau}, m={embedding_dim})"
            description = f"2D reconstruction of the attractor using time-delay embedding with τ={tau} and embedding dimension m={embedding_dim}."
            
        else:
            # For higher dimensions, plot some 2D projections
            def plot_func():
                fig, axs = plt.subplots(1, 2, figsize=(12, 5))
                
                # First projection: dimensions 0 and 1
                axs[0].plot(embedded_data[:, 0], embedded_data[:, 1], lw=0.5)
                axs[0].set_xlabel(f'${original_var}(t)$')
                axs[0].set_ylabel(f'${original_var}(t+{tau})$')
                axs[0].grid(True, alpha=0.3)
                axs[0].set_title("Projection onto dimensions 1-2")
                
                # Second projection: dimensions 0 and 2
                axs[1].plot(embedded_data[:, 0], embedded_data[:, 2], lw=0.5)
                axs[1].set_xlabel(f'${original_var}(t)$')
                axs[1].set_ylabel(f'${original_var}(t+{2*tau})$')
                axs[1].grid(True, alpha=0.3)
                axs[1].set_title("Projection onto dimensions 1-3")
                
                plt.tight_layout()
                
            title = f"Reconstructed Attractor Projections (τ={tau}, m={embedding_dim})"
            description = f"2D projections of the {embedding_dim}-dimensional reconstructed attractor using time-delay embedding with τ={tau}."
        
        return self.save_plot(
            plot_func, 
            f"embedded_attractor_tau{tau}_dim{embedding_dim}", 
            title, 
            description
        )
    
    def plot_mutual_information(self, delays, mi_values, optimal_delay):
        """
        Plot mutual information vs delay and save the result.
        
        Parameters:
            delays (ndarray): Delay values
            mi_values (ndarray): Mutual information values
            optimal_delay (int): The selected optimal delay
            
        Returns:
            str: Path to the saved plot
        """
        def plot_func():
            plt.plot(delays, mi_values)
            plt.axvline(x=optimal_delay, color='r', linestyle='--', 
                       label=f'Optimal delay: {optimal_delay}')
            plt.xlabel('Delay')
            plt.ylabel('Mutual Information')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        title = "Mutual Information vs Delay"
        description = (f"Mutual information between time series and its delayed version as a function of delay. "
                      f"The first minimum at τ={optimal_delay} is selected as the optimal embedding delay.")
            
        return self.save_plot(
            plot_func, 
            "mutual_information", 
            title, 
            description
        )
    
    def plot_correlation_dimension(self, dim_results, embedding_dim):
        """
        Plot the correlation sum and dimension estimation.
        
        Parameters:
            dim_results (dict): Results from dimension estimation
            embedding_dim (int): The embedding dimension used
            
        Returns:
            str: Path to the saved plot
        """
        def plot_func():
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot all points
            ax.scatter(dim_results['log_eps'], dim_results['log_c'], 
                      s=30, alpha=0.7, label='All points')
            
            # Highlight scaling region
            scaling_region = dim_results['scaling_region']
            ax.scatter(dim_results['log_eps'][scaling_region[0]:scaling_region[1]], 
                      dim_results['log_c'][scaling_region[0]:scaling_region[1]], 
                      s=50, color='red', label='Scaling region')
            
            # Plot the fit line
            x_fit = np.linspace(dim_results['scaling_log_eps'][0], 
                              dim_results['scaling_log_eps'][-1], 100)
            y_fit = dim_results['fit_slope'] * x_fit + dim_results['fit_intercept']
            ax.plot(x_fit, y_fit, 'k--', 
                   label=f'Fit: D = {dim_results["dimension"]:.3f}')
            
            ax.set_xlabel('log(ε)')
            ax.set_ylabel('log(C(ε))')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        title = f"Correlation Dimension Estimation (m={embedding_dim})"
        description = (f"Log-log plot of correlation sum C(ε) vs. scale ε for the embedded attractor "
                      f"with embedding dimension m={embedding_dim}. "
                      f"The slope of the linear scaling region gives an estimate of the "
                      f"correlation dimension D = {dim_results['dimension']:.3f}.")
            
        return self.save_plot(
            plot_func, 
            f"correlation_dimension_m{embedding_dim}", 
            title, 
            description
        )
    
    def save_parameters(self, params_dict):
        """
        Save simulation and analysis parameters.
        
        Parameters:
            params_dict (dict): Dictionary of parameters
        """
        params_path = os.path.join(self.run_dir, "parameters.json")
        with open(params_path, 'w') as f:
            json.dump(params_dict, f, indent=4)
            
        # Add parameters to dashboard
        with open(self.dashboard_file, 'a') as f:
            f.write("\n## Parameters\n\n")
            f.write("```json\n")
            f.write(json.dumps(params_dict, indent=4))
            f.write("\n```\n")
    
    def _update_dashboard(self):
        """Update the dashboard markdown file with current plots and info."""
        with open(self.dashboard_file, 'w') as f:
            f.write(f"# Chaotic System Analysis Dashboard\n\n")
            f.write(f"**Run ID:** {self.run_id}\n\n")
            f.write(f"**Date/Time:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Plots\n\n")
            
            for i, plot_info in enumerate(self.plots_info):
                f.write(f"### {i+1}. {plot_info['title']}\n\n")
                f.write(f"{plot_info['description']}\n\n")
                f.write(f"![{plot_info['title']}](plots/{plot_info['filename']}.png)\n\n")
                f.write("---\n\n")
                
    def finalize(self, conclusion_text=None):
        """
        Finalize the run and update the dashboard with conclusions.
        
        Parameters:
            conclusion_text (str): Optional conclusion text
        """
        if conclusion_text:
            with open(self.dashboard_file, 'a') as f:
                f.write("\n## Conclusions\n\n")
                f.write(f"{conclusion_text}\n\n")
        
        # Create a simple index.html that redirects to the dashboard
        index_path = os.path.join(self.run_dir, "index.html")
        with open(index_path, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url='dashboard.md'" />
</head>
<body>
    <p>Redirecting to <a href="dashboard.md">dashboard</a>...</p>
</body>
</html>
""")
        
        print(f"Run completed. Results saved to {self.run_dir}")
        print(f"Dashboard available at {self.dashboard_file}") 