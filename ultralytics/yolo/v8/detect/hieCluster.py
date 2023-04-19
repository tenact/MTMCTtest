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
    Das stellt eine vereinfachte Version des Clusterings dar, es gibt keine richtiges Clustering
    => aber Schritt ist meiner Sicht unnnötig? => Es gibt nur Clustering, wenn ein neues Element entsteht.
    also gibt es Clustering.
    => durch die fehlende Ähnlichkeit gibt es halt weniger Cluster


    Feature-Arrays: Ein Dict mit allen bisherig getrackten Feature-Arrays
    query-Feature: Der neue entdeckte Track, die ResNet Feature-Array (2048) Elemente 1 Dimensional
    query-ID: Die ID des neuen Tracks
    
    Wenn der Cluster leer ist, wird der neue Track als Cluster hinzugefügt
    und die die query-ID als Cluster Label gesetzt.
    Der Centroid ist der neue Track
    Wenn der Cluster nicht leer ist, wird der neue Track mit allen bisherigen Tracks verglichen
    Wenn der neue Track ähnlich zu einem anderen ist, wird der neue Track dem Cluster hinzugefügt
    und die Id des Clusters wird zurückgegeben

    Hinweis:
    Ich brauche keinen vorherigen Cluster-Schritt, da ich die Elemente immer hinzufüge
    und das ab dem ersten Element

    Also erstelle ich für jeden Track einen Cluster 
    und wenn es eine Ähnlichkeit gibt
    neu Berechnung des Centroids und Rückgabe der ID

    

    '''

    # Durchführung des Hierarchical Clusterings
    # Rückgabe der ID der Feature Array, beim match, sonst NONE
    #Nutzung der Query ID als Cluster Label. 
    def clustering_do(self, feature_arrays, query_feature, query_id): 

        print("Aufruf der clustering Methode und laden der Parameter")
     
        # Id die Ausgegeben wird
        reidIdendity = -1


        if len(centroid_dict) == 0:
            centroid_dict[query_id] = query_feature
            cluster_labels[query_id] = query_id
            return -1
    
        print("die länge centroid_dict: " + str(len(centroid_dict)))
        

    
        for key, centroid in centroid_dict.items():
            distance = np.linalg.norm(centroid-query_feature)/ sqrt(2048)

            print(" The distance: " + str(distance))
            if distance >= 0.8:

                reidIdendity = key
                #centroid_dict[f] = np.mean(np.vstack((centroid_dict[f], query_feature)), axis=0)
                centroid_dict[key] = np.mean([centroid, query_feature], axis=0)

                return reidIdendity
        
        centroid_dict[query_id] = query_feature
        cluster_labels[query_id] = query_id



        return reidIdendity


















        '''

        
        #load_clusters()


        print("Aufruf der clustering Methode und laden der Parameter")
     
        # Id die Ausgegeben wird
        reidIdendity = -1

        distance_threshold = 0.8 # oder als Argument übergeben
        
        # Define a custom distance function based on cosine similarity 
        #oder einfach Euclidean distance verwenden
        def cosine_similarity(u, v):
            return 1 - np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

        
        # Initialize the cluster labels, centroids, and clustering model
        #cluster_labels = np.array([0])


        #if not cluster_labels: # wir prüfen, ob der Cluster leer ist oder non existent
        print("cluster ist leeer und ein Element wird hinzugefügt")
        cluster_labels = np.array([query_id]) # hinzufügen der query ID als Cluster Label
        cluster_centroids = np.array([query_feature])

        clustering = AgglomerativeClustering(n_clusters=1, affinity='precomputed', linkage='single')


        if len(cluster_labels) > 1:
            print("Length of cluster_labels: " + str(len(cluster_labels)) + " Length of cluster_centroids: " + str(len(cluster_centroids)) + " Length of clustering: " + str(len(clustering)))
        
        #feature_arrays = feature_arrays.reshape((1, 2048))
        # Compute the pairwise distances between feature arrays and existing centroids
        #distances = cdist(feature_arrays,   cluster_centroids, metric=cosine_similarity)


        

        if not clustering.children_.any():
            new_cluster_label = max(cluster_labels) + 1
            cluster_labels = np.append(cluster_labels, new_cluster_label)
            cluster_centroids = np.vstack((cluster_centroids, query_feature))

        # Iterate over feature arrays and update clusters
        for i in range(len(feature_arrays)):
            query_distance = distances[i, 0]  # distance to the existing centroid
            nearest_cluster_idx = np.argmin(distances[i, :])  # index of the nearest centroid

            if query_distance >= distance_threshold:
                # If the query element is larger than the centroid, add it to the cluster

                reidIdendity = cluster_labels[nearest_cluster_idx]
                cluster_labels = np.append(cluster_labels, nearest_cluster_idx)
                cluster_centroids[nearest_cluster_idx] = np.mean(np.vstack((cluster_centroids[nearest_cluster_idx], feature_arrays[i])), axis=0)
            else:
                # If the query element is dissimilar to all centroids, create a new cluster with the query element
                new_cluster_label = max(cluster_labels) + 1
                cluster_labels = np.append(cluster_labels, new_cluster_label)
                cluster_centroids = np.vstack((cluster_centroids, feature_arrays[i]))

        # Perform hierarchical clustering with single linkage using the updated cluster centroids
        clustering.n_clusters = len(cluster_centroids)
        clustering.fit(distances)


        return reidIdendity


        toAdd = False

        for i in range(len(cluster_centroids)):
            if cdist(query_feature, cluster_centroids[i], metric=cosine_similarity) > distance_threshold:
                # If the query element is larger than the centroid, add it to the cluster
                cluster_labels = np.append(cluster_labels, query_id)
                cluster_centroids[i] = np.mean(np.vstack((cluster_centroids[i], query_feature)), axis=0)
                toAdd = True
                break
            if toAdd:
                #hinzufügen zu einem eigenen Cluster
                new_cluster_label = max(cluster_labels) + 1
                cluster_labels = np.append(cluster_labels, new_cluster_label)
                cluster_centroids = np.vstack((cluster_centroids, feature_arrays[i]))



        clustering.n_clusters = len(cluster_centroids)
        clustering.fit(distances)


        return reidIdendity
        

        
        # If there are no hierarchical clusters, create one with the query element
        if not clustering.children_.any():
            new_cluster_label = max(cluster_labels) + 1
            cluster_labels = np.append(cluster_labels, new_cluster_label)
            cluster_centroids = np.vstack((cluster_centroids, query_feature))
        '''