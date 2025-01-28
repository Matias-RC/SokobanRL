import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

class ModifiedDBSCAN:
    def __init__(self, epsilon=0.5, min_samples=4, fine_tune=False, max_iterations=50, epsilon_tolerance=1e-4):
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.core_points = set()
        self.cluster_labels = {}
        self.clusters = defaultdict(list)
        self.fine_tune = fine_tune
        self.max_iterations = max_iterations
        self.epsilon_tolerance = epsilon_tolerance

    def _find_neighbors(self, vector, data):
        """Find all neighbors of a vector within epsilon distance."""
        similarities = cosine_similarity([vector], data)[0]
        return np.where(similarities >= self.epsilon)[0]

    def _identify_cores(self, data):
        """Identify core points in the first group of vectors."""
        self.core_points.clear()
        for idx, vector in enumerate(data):
            neighbors = self._find_neighbors(vector, data)
            if len(neighbors) >= self.min_samples:
                self.core_points.add(idx)

    def _expand_cluster(self, core_idx, cluster_id, data, visited):
        """Expand a cluster starting from a core point."""
        queue = [core_idx]
        while queue:
            point_idx = queue.pop(0)

            if point_idx in visited:
                continue

            visited.add(point_idx)
            if point_idx not in self.cluster_labels:
                self.cluster_labels[point_idx] = cluster_id
                self.clusters[cluster_id].append(point_idx)

            # Only expand clusters from core points
            if point_idx in self.core_points:
                neighbors = self._find_neighbors(data[point_idx], data)
                for neighbor_idx in neighbors:
                    if neighbor_idx not in visited:
                        queue.append(neighbor_idx)

    def _get_cluster_count(self, data):
        """Count the number of clusters based on current core points."""
        visited = set()
        cluster_id = 0
        self.cluster_labels.clear()
        self.clusters.clear()

        for core_idx in self.core_points:
            if core_idx not in visited:
                self._expand_cluster(core_idx, cluster_id, data, visited)
                cluster_id += 1

        return cluster_id
    def _fine_tune_epsilon(self, data, target_clusters):
        """Fine-tune epsilon to achieve the target number of clusters."""
        # Start from the initial epsilon
        base_epsilon = self.epsilon
        best_epsilon = base_epsilon
        best_cluster_count = float("inf")
        min_difference = float("inf")

        # Generate candidate epsilon values
        steps = 50
        epsilon_range = np.linspace(base_epsilon - 0.1, base_epsilon + 0.1, steps)

        for candidate_epsilon in epsilon_range:
            self.epsilon = candidate_epsilon
            self._identify_cores(data)

            # Calculate clusters for the current epsilon
            cluster_count = self._get_cluster_count(data)

            # Evaluate how close we are to the target
            difference = abs(cluster_count - target_clusters)

            if difference < min_difference:
                best_epsilon = candidate_epsilon
                best_cluster_count = cluster_count
                min_difference = difference

            # Stop if we match the target exactly
            if difference == 0:
                break

        # Update epsilon to the best value found
        self.epsilon = best_epsilon
        print(f"Fine-tuned epsilon: {self.epsilon:.5f}, clusters: {best_cluster_count}")



    def fit(self, group1, group2, target_clusters=None):
        """Fit the algorithm to two groups of vectors."""
        # Fine-tune epsilon if enabled and target clusters are provided
        if self.fine_tune and target_clusters is not None:
            self._fine_tune_epsilon(group1, target_clusters)

        # Identify core points from group1
        self._identify_cores(group1)

        # Combine both groups
        combined_data = np.vstack([group1, group2])

        # Form clusters on combined data
        visited = set()
        cluster_id = 0
        for core_idx in self.core_points:
            if core_idx not in visited:
                self._expand_cluster(core_idx, cluster_id, combined_data, visited)
                cluster_id += 1

        return self
    def get_clusters(self):
        """Retrieve clusters as lists of vector indices."""
        return {k: sorted(v) for k, v in self.clusters.items()}



"""Example Usage"""
# Simulated high-dimensional vectors
np.random.seed(595)
group1 = np.random.rand(500, 20)  # First group of vectors
group2 = np.random.rand(1000, 20)   # Second group of vectors

# Create and fit the modified DBSCAN
dbscan = ModifiedDBSCAN(epsilon=0.9, min_samples=4, fine_tune=True)
dbscan.fit(group1, group2, target_clusters=5)

# Output the resulting clusters
clusters = dbscan.get_clusters()
for cluster_id, indices in clusters.items():
    print(f"Cluster {cluster_id}: {indices}")
