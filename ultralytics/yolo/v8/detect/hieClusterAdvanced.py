from math import sqrt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
import pickle



 
def save_clusters(self, file_prefix):
        with open(file_prefix + '_labels.pkl', 'wb') as f:
            pickle.dump(self.cluster_labels, f)
        with open(file_prefix + '_centroids.pkl', 'wb') as f:
            pickle.dump(self.cluster_centroids, f)
        with open(file_prefix + '_clusters.pkl', 'wb') as f:
            pickle.dump(self.clustering, f)
        print("Cluster labels, centroids, and clusters saved successfully!")
    
    # Load cluster labels, centroids, and clusters from separate files
def load_clusters(self, file_prefix):
    with open(file_prefix + '_labels.pkl', 'rb') as f:
        self.cluster_labels = pickle.load(f)
    with open(file_prefix + '_centroids.pkl', 'rb') as f:
        self.cluster_centroids = pickle.load(f)
    with open(file_prefix + '_clusters.pkl', 'rb') as f:
        self.clustering = pickle.load(f)
    print("Cluster labels, centroids, and clusters loaded successfully!")


cluster_labels = {} # read from file
cluster_centroids = {} # read from file
clustering = [] # read from file
centroid_dict = {}


class HIECLU:

    '''
    Feature-Arrays: Ein Dict mit allen bisherig getrackten Feature-Arrays
    query-Feature: Der neue entdeckte Track, die ResNet Feature-Array (2048) Elemente 1 Dimensional
    query-ID: Die ID des neuen Tracks
    
    Wenn der Cluster leer ist, wird der neue Track als Cluster hinzugefügt
    und die die query-ID als Cluster Label gesetzt.
    Der Centroid ist der neue Track
    Wenn der Cluster nicht leer ist, wird der neue Track mit allen bisherigen Tracks verglichen
    Wenn der neue Track ähnlich zu einem anderen ist, wird der neue Track dem Cluster hinzugefügt
    und die Id des Clusters wird zurückgegeben

    

    '''

    # Durchführung des Hierarchical Clusterings
    # Rückgabe der ID der Feature Array, beim match, sonst NONE
    #Nutzung der Query ID als Cluster Label. 
    def clustering_do(self, feature_arrays, query_feature, query_id): 

    
        return 1


    

    

    def agglomerative_clustering_single_linkage(features_dict, query_id, query_feature):
        # Initialize clusters with each feature as its own cluster
        clusters = {key: [key] for key in features_dict.keys()}
        
        # Initialize centroids as feature arrays
        centroids = {key: feature for key, feature in features_dict.items()}
        
        # Loop until there is only one cluster left
        while len(clusters) > 1:
            # Find the closest two clusters
            min_distance = float('inf')
            closest_clusters = None
            for i, (id_i, cluster_i) in enumerate(clusters.items()):
                for j, (id_j, cluster_j) in enumerate(clusters.items()):
                    if i < j:
                        distance = euclidean_distance(centroids[id_i], centroids[id_j])
                        if distance < min_distance:
                            min_distance = distance
                            closest_clusters = (id_i, id_j)
            
            # Merge the closest two clusters into a new cluster
            id_new = closest_clusters[0] + '+' + closest_clusters[1]
            cluster_new = clusters[closest_clusters[0]] + clusters[closest_clusters[1]]
            clusters[id_new] = cluster_new
            
            # Update the centroids dictionary
            centroid_new = np.mean([features_dict[id] for id in cluster_new], axis=0)
            centroids[id_new] = centroid_new
            
            # Remove the old clusters from the dictionary
            del clusters[closest_clusters[0]]
            del clusters[closest_clusters[1]]
        
        # Find the cluster containing the query feature
        query_cluster = None
        for cluster in clusters.values():
            if query_id in cluster:
                query_cluster = cluster
                break
        
        # Return the cluster containing the query feature and the centroids dictionary
        return query_cluster, centroids
