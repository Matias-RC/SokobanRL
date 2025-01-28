import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

class ModifiedDBSCAN:
    def __init__(self, epsilon=0.5, min_samples=4):
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.core_points = set()
        self.cluster_labels = {}
        self.clusters = defaultdict(list)

    def _find_neighbors(self, vector, data):
        """Find all neighbors of a vector within epsilon distance."""
        similarities = cosine_similarity([vector], data)[0]
        return np.where(similarities >= self.epsilon)[0]

    def _identify_cores(self, data):
        """Identify core points in the first group of vectors."""
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
            self.cluster_labels[point_idx] = cluster_id
            self.clusters[cluster_id].append(point_idx)

            # Only core points can expand the cluster
            if point_idx in self.core_points:
                neighbors = self._find_neighbors(data[point_idx], data)
                for neighbor_idx in neighbors:
                    if neighbor_idx not in visited:
                        queue.append(neighbor_idx)

    def fit(self, group1, group2):
        """Fit the algorithm to two groups of vectors."""
        # Process the first group to identify cores
        self._identify_cores(group1)

        # Form clusters based on cores in group1
        visited = set()
        cluster_id = 0
        for core_idx in self.core_points:
            if core_idx not in visited:
                self._expand_cluster(core_idx, cluster_id, group1, visited)
                cluster_id += 1

        # Incorporate group2 into the clusters
        for idx, vector in enumerate(group2):
            max_similarity = 0
            assigned_cluster = None

            # Assign to the closest cluster if within epsilon
            for cluster_id, core_indices in self.clusters.items():
                core_vectors = group1[core_indices]
                similarities = cosine_similarity([vector], core_vectors)[0]
                if np.max(similarities) >= self.epsilon:
                    if np.max(similarities) > max_similarity:
                        max_similarity = np.max(similarities)
                        assigned_cluster = cluster_id

            # Only assign if it connects to a cluster
            if assigned_cluster is not None:
                self.cluster_labels[len(group1) + idx] = assigned_cluster
                self.clusters[assigned_cluster].append(len(group1) + idx)

        return self

    def get_clusters(self):
        """Retrieve clusters as lists of vector indices."""
        return {k: sorted(v) for k, v in self.clusters.items()}
    
# Simulated high-dimensional vectors
np.random.seed(42)
group1 = np.random.rand(10, 50)  # First group of vectors
group2 = np.random.rand(5, 50)   # Second group of vectors

# Create and fit the modified DBSCAN
dbscan = ModifiedDBSCAN(epsilon=0.7, min_samples=4)
dbscan.fit(group1, group2)

# Output the resulting clusters
clusters = dbscan.get_clusters()
for cluster_id, indices in clusters.items():
    print(f"Cluster {cluster_id}: {indices}")
