import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class GaussianMixtureModel(object):
    def __init__(self, num_clusters: int, num_dimensions: int):
        '''
        num_clusters: Number of clusters
        num_dimensions: Number of attributes per sample
        '''
        self.K = num_clusters
        # Initialize parameters
        self.mixing_coefficients = np.random.rand(num_clusters)
        self.mixing_coefficients = self.mixing_coefficients / self.mixing_coefficients.sum()  # Ensure sum of all mixing coefficients is 1
        self.means = np.random.rand(num_clusters, num_dimensions)
        self.covariances = np.empty((num_clusters, num_dimensions, num_dimensions))
        for i in range(num_clusters):
            self.covariances[i] = np.eye(num_dimensions) * np.random.rand(1) * num_clusters

    def fit(self, data: np.ndarray, max_iterations: int = 100):
        '''
        data: Data matrix, each row is a sample, shape = (N, num_dimensions)
        max_iterations: Maximum number of iterations for EM algorithm
        '''
        for _ in range(max_iterations):
            densities = np.empty((len(data), self.K))
            for i in range(self.K):
                # Generate K probability density functions and calculate the density for all samples
                norm = stats.multivariate_normal(self.means[i], self.covariances[i])
                densities[:, i] = norm.pdf(data)
            # Calculate the posterior for all samples belonging to each cluster
            posterior = densities * self.mixing_coefficients
            posterior = posterior / posterior.sum(axis=1, keepdims=True)
            # Calculate the parameter values for the next iteration
            p_hat = posterior.sum(axis=0)
            mean_hat = np.tensordot(posterior, data, axes=[0, 0])
            # Calculate the covariance
            cov_hat = np.empty(self.covariances.shape)
            for i in range(self.K):
                tmp = data - self.means[i]
                cov_hat[i] = np.dot(tmp.T * posterior[:, i], tmp) / p_hat[i]
            # Update parameters
            self.covariances = cov_hat
            self.means = mean_hat / p_hat.reshape(-1, 1)
            self.mixing_coefficients = p_hat / len(data)

        print("Final mixing coefficients:", self.mixing_coefficients)
        print("Final means:", self.means)
        print("Final covariances:", self.covariances)

# Generate test data
np.random.seed(42)
num_samples = 500
num_dimensions = 2  # Dimension of the data

# Generate data points from 3 Gaussian distributions
mean1, cov1 = [2, 3], [[1, 0.5], [0.5, 1]]
mean2, cov2 = [-1, -2], [[0.5, -0.3], [-0.3, 0.5]]
mean3, cov3 = [5, -5], [[1, 0], [0, 1]]

data1 = np.random.multivariate_normal(mean1, cov1, num_samples // 3)
data2 = np.random.multivariate_normal(mean2, cov2, num_samples // 3)
data3 = np.random.multivariate_normal(mean3, cov3, num_samples // 3)

data = np.vstack((data1, data2, data3))

# Visualize the generated data
plt.scatter(data[:, 0], data[:, 1], s=5)
plt.title('Generated Test Data')
plt.show()

# Create and fit GMM
num_clusters = 3  # Number of clusters
gmm = GaussianMixtureModel(num_clusters, num_dimensions)
gmm.fit(data)

# Print fitted parameters
print("Final mixing coefficients:", gmm.mixing_coefficients)
print("Final means:", gmm.means)
print("Final covariances:", gmm.covariances)
