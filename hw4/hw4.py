import geopandas
import numpy as np
from matplotlib import pyplot as plt
import csv
from scipy.cluster.hierarchy import dendrogram
from math import factorial

def world_map(Z, names, K_clusters):
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    world['name'] = world['name'].str.strip()
    names = [name.strip() for name in names]

    world['cluster'] = np.nan

    n = len(names)
    clusters = {j: [j] for j in range(n)}

    for step in range(n-K_clusters):
        cluster1 = Z[step][0]
        cluster2 = Z[step][1]

        # Create new cluster id as n + step
        new_cluster_id = n + step

        # Merge clusters
        clusters[new_cluster_id] = clusters.pop(cluster1) + clusters.pop(cluster2)

    # Assign cluster labels to countries in the world dataset
    for i, value in enumerate(clusters.values()):
        for val in value:
            world.loc[world['name'] == names[val], 'cluster'] = i

    # Plot the map
    world.plot(column='cluster', legend=True, figsize=(15, 10), missing_kwds={
        "color": "lightgrey",  # Set the color of countries without clusters
        "label": "Other countries"
    })

    # Show the plot
    plt.show()

def load_data(filepath : str) -> list:
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)
        return data

def calc_features(row : dict) -> np.array:
    x1 = float(row['Population'])
    x2 = float(row['Net migration'])
    x3 = float(row['GDP ($ per capita)'])
    x4 = float(row['Literacy (%)'])
    x5 = float(row['Phones (per 1000)'])
    x6 = float(row['Infant mortality (per 1000 births)'])

    feature_array = np.array([x1, x2, x3, x4, x5, x6], dtype = np.float64)
    # feature_array = np.reshape(feature_array,(6,))
    # array is of shape (6,)

    return feature_array

def hac(features : np.array) -> np.array:
    # distance matrix - where distance_matrix[i, j] is the distance between the countries at index i and j
    # computes Euclideance distance between all pairs of countries (feature vectors)

    # initialize number of input features and the distance matrix

    # a linkage matrix to store the result of the single-linkage hierarchical agglomerative clustering
    # Z[i] will tell us which clusters were merged in the i-th iteration (step)
    # Z[i] = [cluster1, cluster2, distance, size]
    
    n = len(features) # Array m x n
    distance_matrix = np.zeros((2*n-1, 2*n-1))
    distance_matrix = np.where(distance_matrix < 0, distance_matrix, np.inf)

    for i in range(n):
        for j in range(n):
            if i <= j:
                continue
            # if i < j:
            #     distance_matrix[i, j] = np.inf # set the upper triangle to np.inf to avoid double calculation
            # elif i == j:
            #     distance_matrix[i, j] = np.inf # set the diagonal to np.inf to avoid merging the same cluster
            elif i > j:
                _i = features[i]
                _j = features[j]
                assert type(_i) == np.ndarray and type(_j) == np.ndarray, f"Features must be numpy arrays. {i} and {j} are not numpy arrays."
                distance_matrix[i, j] = np.linalg.norm(features[i] - features[j]) # Euclidean distance, always positive.
    
    # initialize the linkage matrix
    Z = np.zeros((n - 1, 4))
    # initially, each country is its own cluster
    clusters = {i: [i] for i in range(n)}

    # k iterations (n times)
    for k in range(n-1):
        print(factorial(n-k)/(2*factorial(n-k-2)), ":", np.count_nonzero(distance_matrix != np.inf))
        # get min distance
        merge_idx = n+k
        min_distance = np.min(distance_matrix)
        assert min_distance != np.inf, "Minimum distance is infinity. Check the distance matrix."
        assert min_distance >= 0, "Minimum distance is negative. Check the distance matrix."
        assert min_distance in distance_matrix, "Minimum distance is not in the distance matrix. Check the distance matrix."
        merging = (-1,-1)

        ### remove
        print(n+k, clusters)
        tmp = 0
        for row in distance_matrix:
            if tmp == n+k:
                break
            for col in row:
                if col == np.inf:
                    print(0, end = " ")
                else:
                    print(1, end = " ")
            print("|", tmp, end = "\n") 
            tmp+= 1
        for i in range(2*n-1):
            print(i, end = " "),
        print("\n")
        ### remove

        # get the cluster idxes of min distance
        for i in range(merge_idx):
            for j in range(merge_idx): # clusters.keys():
                if i <= j:
                    continue
                elif distance_matrix[i, j] == min_distance:
                    merging = (i, j)
                    distance_matrix[i, j] = np.inf # set the distance to np.inf to avoid re-merging the same clusters
                    break
       
        # update the distance matrix
        z0 = min(merging)
        z1 = max(merging)
        clusters[merge_idx] = clusters.pop(z0) + clusters.pop(z1)

        for j in range(0, merge_idx):
            distance_matrix[merge_idx, j] = min(distance_matrix[z0, j], distance_matrix[z1, j])

        for i in range(0, merge_idx):
            for j in range(0, merge_idx):
                if i <= j:
                    continue
                elif (i == z0 or i == z1) or (j == z0 or j == z1):
                    distance_matrix[i, j] = np.inf

        # update the linkage matrix
        Z[k] = [z0, z1, min_distance, len(clusters[merge_idx])]

        # print(Z) # remove
    return Z # return the linkage matrix

def fig_hac(Z : np.array, names : list):
    # Plot the dendrogram
    fig = plt.figure() # figsize=(50, 80)
    #plt.title("N = " + str(len(names)))
    plt.tight_layout()
    plt.show()
    # creates a dendrogram based on the given linkage matrix
    dendrogram(Z, labels = names, leaf_rotation = 90)

    return fig

def normalize_features(features : list) -> list:
    # normalize the data to ensure that the clustering is not biased by the scale of the features (e.g., population vs. GDP)
    # compute mean and standard deviation of all features and normalize them.

    features = np.array(features)
    
    #feature_means = np.mean(features, axis = 0)
    #feature_std_devs = np.std(features, axis = 0)
    #normalized_features = (features - feature_means) / feature_std_devs

    col_mins = np.min(features, axis = 0)
    col_maxs = np.max(features, axis = 0)

    normalized_features = (features - col_mins) / (col_maxs - col_mins)
    normalized_features = [row for row in normalized_features]

    return normalized_features

if __name__ == "__main__":
    data = load_data("countries.csv")
    country_names = [row["Country"] for row in data]
    features = [calc_features(row) for row in data]
    features_normalized = normalize_features(features)

    n = 6
    # print(f"Testing for n = {n}")
    # for row in data[-n:]:
    #     print(row)
    Z_raw = hac(features[-n:])
    #Z_normalized = hac(features_normalized[:n])
    
    fig_hac(Z_raw, country_names[-n:])
    #fig_hac(Z_normalized, country_names[:n]) 