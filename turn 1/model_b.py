import unittest
from unittest.mock import patch
import matplotlib.pyplot as plt
import numpy as np
from vision_viz import plot_histogram

# Dummy data for testing
dummy_data = np.random.randint(0, 100, size=100)

class TestPlotHistogram(unittest.TestCase):
    @patch('matplotlib.pyplot.show')
    def test_plot_histogram_output(self, mock_show):
        """
        Test if plot_histogram generates a histogram with the correct output.
        """
        plot_histogram(dummy_data)
        mock_show.assert_called_once()  # Ensure show() is called once

    @patch('matplotlib.pyplot.figure')
    def test_plot_histogram_figure_size(self, mock_figure):
        """
        Test if plot_histogram sets the figure size correctly.
        """
        plot_histogram(dummy_data, figsize=(10, 6))
        mock_figure.assert_called_once_with(figsize=(10, 6))  # Ensure figsize is set

    def test_plot_histogram_bins(self):
        """
        Test if plot_histogram uses the correct number of bins.
        """
        bins = 20
        _, axes = plt.subplots()
        plot_histogram(dummy_data, bins=bins, ax=axes)
        hist, _ = np.histogram(dummy_data, bins=bins)
        self.assertEqual(len(axes.patches), len(hist))  # Ensure number of bars matches bins

    def test_plot_histogram_compliance_with_spec(self):
        """
        Test if plot_histogram adheres to vision system specification.
        """
        # Assuming the specification requires the histogram to be normalized
        _, axes = plt.subplots()
        plot_histogram(dummy_data, ax=axes)
        hist, _ = np.histogram(dummy_data, density=True)
        np.testing.assert_allclose(axes.patches[0].get_height(), hist[0], rtol=1e-05)

if __name__ == '__main__':
    unittest.main()
