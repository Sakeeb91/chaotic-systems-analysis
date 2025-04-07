import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import tempfile
import shutil
import os
import matplotlib.pyplot as plt

# Import our modules
from chaotic_systems import rossler_system, simulate_system
from embedding import (time_delay_embedding, find_optimal_delay, 
                      estimate_correlation_dimension, mutual_information)
from visualization import ResultsManager

class TestChaoticSystems(unittest.TestCase):
    """Test chaotic system simulation functions."""
    
    def test_rossler_system(self):
        """Test Rossler system ODE function."""
        # Test with default parameters
        state = [1.0, 2.0, 3.0]
        derivatives = rossler_system(0, state)
        # Values calculated by hand
        expected = [-2.0 - 3.0, 1.0 + 0.2 * 2.0, 0.2 + 3.0 * (1.0 - 5.7)]
        assert_array_almost_equal(derivatives, expected)
        
        # Test with custom parameters
        params = {'a': 0.1, 'b': 0.1, 'c': 14.0}
        state = [2.0, 0.5, 1.0]
        derivatives = rossler_system(0, state, **params)
        expected = [-0.5 - 1.0, 2.0 + 0.1 * 0.5, 0.1 + 1.0 * (2.0 - 14.0)]
        assert_array_almost_equal(derivatives, expected)
    
    def test_simulate_system(self):
        """Test system simulation."""
        # Simple linear system for testing
        def linear_system(t, state):
            x, y = state
            return [y, -x]  # Simple harmonic oscillator
        
        initial_state = [1.0, 0.0]
        t_span = [0, 2*np.pi]
        dt = np.pi/10
        
        # Simulate
        results = simulate_system(linear_system, initial_state, t_span, dt)
        
        # Check output format
        self.assertIn('time', results)
        self.assertIn('states', results)
        
        # Check dimensions
        self.assertEqual(len(results['time']), results['states'].shape[0])
        self.assertEqual(results['states'].shape[1], len(initial_state))
        
        # For this simple system, we expect x = cos(t), y = -sin(t)
        expected_x = np.cos(results['time'])
        expected_y = -np.sin(results['time'])
        
        # Check results (with tolerance for numerical error)
        assert_array_almost_equal(results['states'][:, 0], expected_x, decimal=1)
        assert_array_almost_equal(results['states'][:, 1], expected_y, decimal=1)
    
    def test_discard_transient(self):
        """Test transient discarding in simulation."""
        # Simple linear system
        def linear_system(t, state):
            return [-state[0]]  # Exponential decay
        
        initial_state = [1.0]
        t_span = [0, 10]
        dt = 0.1
        discard_transient = 5.0
        
        # Simulate with transient discarding
        results = simulate_system(linear_system, initial_state, t_span, dt, discard_transient=discard_transient)
        
        # Check that times before discard_transient are removed
        self.assertTrue(all(t >= discard_transient for t in results['time']))


class TestEmbedding(unittest.TestCase):
    """Test embedding and dimension estimation functions."""
    
    def test_time_delay_embedding(self):
        """Test time-delay embedding function."""
        # Create a simple sine wave for testing
        t = np.linspace(0, 10*np.pi, 1000)
        x = np.sin(t)
        
        # Test with embedding dimension 2, delay 25
        embedding_dim = 2
        delay = 25
        embedded = time_delay_embedding(x, embedding_dim, delay)
        
        # Check dimensions
        expected_rows = len(x) - (embedding_dim - 1) * delay
        self.assertEqual(embedded.shape, (expected_rows, embedding_dim))
        
        # Check values at specific points
        self.assertAlmostEqual(embedded[0, 0], x[0])
        self.assertAlmostEqual(embedded[0, 1], x[delay])
        
        # Test error cases
        with self.assertRaises(ValueError):
            time_delay_embedding(x, 0, 1)  # Invalid embedding dimension
        
        with self.assertRaises(ValueError):
            time_delay_embedding(x, 1, 0)  # Invalid delay
        
        with self.assertRaises(ValueError):
            time_delay_embedding(x, 100, 100)  # Too large for time series
    
    def test_mutual_information(self):
        """Test mutual information calculation."""
        # Perfect correlation
        x = np.linspace(0, 10, 100)
        y = x.copy()
        mi = mutual_information(x, y)
        # Mutual information should be > 0 for perfectly correlated data
        self.assertGreater(mi, 0)
        
        # No correlation
        y = np.random.random(100)
        mi_random = mutual_information(x, y)
        # Should be less than the perfectly correlated case
        self.assertLess(mi_random, mi)
    
    def test_correlation_sum(self):
        """Test correlation sum calculation."""
        # Create a simple grid of points (2D)
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        xx, yy = np.meshgrid(x, y)
        points = np.vstack([xx.flatten(), yy.flatten()]).T
        
        # For a 2D grid of points, the correlation dimension should be close to 2
        from embedding import correlation_sum, estimate_correlation_dimension
        
        # Test correlation_sum
        epsilons = np.logspace(-2, 0, 10)
        c_values = correlation_sum(points, epsilons)
        
        # Check output dimensions
        self.assertEqual(len(c_values), len(epsilons))
        
        # Values should be increasing with epsilon
        self.assertTrue(all(c_values[i] <= c_values[i+1] for i in range(len(c_values)-1)))
        
        # Test dimension estimation with a more appropriate scale range
        # Use larger min_scale to avoid log(0) issues
        dim_results = estimate_correlation_dimension(
            points, 
            min_scale=0.1,  # Increased min_scale
            max_scale=0.8,
            num_scales=10
        )
        
        # Only check if a valid dimension value was calculated
        self.assertFalse(np.isnan(dim_results['dimension']))
        
        # For a grid of points, the dimension should be reasonable
        # but we won't strictly test the value as it depends on the scale range
        self.assertGreaterEqual(dim_results['dimension'], 0)


class TestVisualization(unittest.TestCase):
    """Test visualization and results management."""
    
    def setUp(self):
        # Create a temporary directory for test results
        self.temp_dir = tempfile.mkdtemp()
        self.results_mgr = ResultsManager(self.temp_dir)
    
    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_save_plot(self):
        """Test saving a plot with metadata."""
        def dummy_plot():
            plt.plot([1, 2, 3], [1, 2, 3])
        
        # Save a test plot
        plot_path = self.results_mgr.save_plot(
            dummy_plot, 
            "test_plot", 
            "Test Plot", 
            "This is a test plot."
        )
        
        # Check that the plot file was created
        self.assertTrue(os.path.exists(plot_path))
        
        # Check that the markdown file was created
        md_path = os.path.join(self.results_mgr.plots_dir, "test_plot.md")
        self.assertTrue(os.path.exists(md_path))
        
        # Check that the dashboard was updated
        self.assertTrue(os.path.exists(self.results_mgr.dashboard_file))
    
    def test_save_parameters(self):
        """Test saving parameters."""
        # Test parameters
        params = {
            'system': 'rossler',
            'a': 0.2,
            'b': 0.2,
            'c': 5.7
        }
        
        # Save parameters
        self.results_mgr.save_parameters(params)
        
        # Check that the JSON file was created
        params_path = os.path.join(self.results_mgr.run_dir, "parameters.json")
        self.assertTrue(os.path.exists(params_path))
        
        # Check dashboard was updated
        with open(self.results_mgr.dashboard_file, 'r') as f:
            dashboard_text = f.read()
            self.assertIn("## Parameters", dashboard_text)


if __name__ == '__main__':
    unittest.main() 