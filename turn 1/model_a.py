import unittest
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import io


class TestVisualizationCompliance(unittest.TestCase):

    def setUp(self):
        # Dummy data for testing
        self.data = np.random.rand(100, 2)
        self.expected_image_path = 'baseline_image.png'  # Replace this with the actual path

    def test_matplotlib_line_plot_compliance(self):
        """
        Test Matplotlib line plot compliance
        """
        plt.figure()
        plt.plot(self.data[:, 0], self.data[:, 1])
        plt.title('Line Plot')
        plt.xlabel('X')
        plt.ylabel('Y')

        # Generate buffer for in-memory image comparison
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)

        generated_image = Image.open(buffer)
        baseline_image = Image.open(self.expected_image_path)

        # Perform image comparison (Using MSE as an example metric)
        mse = np.mean((np.array(generated_image) - np.array(baseline_image)) ** 2)
        tolerance = 0.01  # Set an appropriate tolerance
        self.assertLess(mse, tolerance, f"MSE exceeds tolerance. MSE: {mse}, Tolerance: {tolerance}")

    def test_seaborn_histogram_compliance(self):
        """
        Test Seaborn histogram compliance
        """
        ax = sns.histplot(self.data[:, 0], bins=20)
        ax.set_title('Histogram')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

        # Generate buffer and perform comparison as in previous test
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
    # rest of the code remains the same


if __name__ == '__main__':
    unittest.main()
