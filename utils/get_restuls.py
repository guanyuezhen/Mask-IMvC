import numpy as np
from sklearn.cluster import KMeans
from utils.clustering_performance import clusteringMetrics

def get_clustering_performance(features, y_label, cluster_num, random_numbers_for_kmeans):
    # Initialize lists to store performance metrics
    metrics = {
        "ACC": [], "NMI": [], "Purity": [],
        "ARI": [], "Fscore": [], "Precision": [], "Recall": []
    }
    for random_state in random_numbers_for_kmeans:
        # Create and fit KMeans model
        kmeans = KMeans(n_clusters=cluster_num, n_init=10, random_state=random_state)
        y_predict = kmeans.fit_predict(features)
        # Calculate clustering metrics
        ACC, NMI, Purity, ARI, Fscore, Precision, Recall = clusteringMetrics(y_label, y_predict)
        # Append metrics to respective lists
        metrics["ACC"].append(ACC)
        metrics["NMI"].append(NMI)
        metrics["Purity"].append(Purity)
        metrics["ARI"].append(ARI)
        metrics["Fscore"].append(Fscore)
        metrics["Precision"].append(Precision)
        metrics["Recall"].append(Recall)
    # Calculate average performance metrics
    average_metrics = {key: np.mean(values) * 100 for key, values in metrics.items()}
    std_metrics = {key: np.std(values) * 100 for key, values in metrics.items()}

    return average_metrics