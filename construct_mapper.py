#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:26:47 2024

@author: meganfairchild

This script is for MAPPER: dimension reduction and clustering analysis. 

You first need to install kmapper using: pip install kmapper 
then you can run the script. 
"""

import kmapper as km
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from kepler_mapper import KeplerMapper
import networkx as nx
import matplotlib.pyplot as plt 
from scipy.spatial.distance import pdist, squareform
import pandas as pd


def construct_mapper_graph_3D(data_array, num_intervals, overlap_frac):

    """
    Constructs a Mapper graph for the given data using dimension reduction and clustering.
    
    Parameters:
    - data_array: numpy array of data points.
    - num_intervals: Number of intervals to divide the range of the filter function.
    - overlap_frac: Fractional overlap between consecutive intervals.
    """

    try:
        if data_array.size == 0:
            raise ValueError("Input data array is empty.")
        
        # Initialize
        mapper = km.KeplerMapper(verbose=1)
        
        # Step 1: Dimension reduction using PCA and DB scan
        pca = PCA(n_components=5)  # Reduce the data to 3 principal components.
        reduced_data = pca.fit_transform(data_array)  # Apply PCA on the input data.
        
        """
        What are the 3 columns used for dimensionality? We can use the pce_contributions to determine how much each column is 
        contributing to the clustering. 

        """
        #first convery the numPy to be able to use Pandas 
        feature_names = [f"Column {i}" for i in range(data_array.shape[1])]  
        X = pd.DataFrame(data_array, columns=feature_names)

        # Get contribution of each feature
        pca_contributions = pd.DataFrame(pca.components_, columns=X.columns, index=[f"PC{i+1}" for i in range(pca.n_components_)])
    
        # Show the top contributing features for each principal component
        print(pca_contributions.abs().idxmax(axis=1))

        """
        The following code is used to check distances from the original space (as in, the ones used to construct the Vietoris Rips complex)
        and compares them to the distances used for the clustering algorithm from PCA. 
        """
        original_distances = squareform(pdist(data_array))  # Distances in original space
        pca_distances = squareform(pdist(reduced_data))  # Distances in PCA space     
        print(f"Original distance range: {original_distances.min()} to {original_distances.max()}")
        print(f"PCA distance range: {pca_distances.min()} to {pca_distances.max()}")
        
        
        """
        Now we use DB scan to form the clusters. 
        """
        dbscan = DBSCAN(eps=2, min_samples=15) #epsilon is distance between points, min_samples is the minimum number of points required to form a dense region 
        #(i.e., a cluster). A point is considered "core" if it has at least min_samples points (including itself) within its eps-radius.
        clusters = dbscan.fit_predict(reduced_data)
        
        # Step 2: Apply Mapper algorithm
        #mapper = Mapper()  # Initialize the Mapper object from scikit-tda.
        #projected_data = mapper.fit_transform(data_array, projection=[0,1]) # X-Y axis
        
        # Create a cover with 10 elements
        #cover = km.Cover(n_cubes=10)
        cover = km.Cover(n_cubes=num_intervals, perc_overlap=overlap_frac)


        # Create dictionary called 'graph' with nodes, edges and meta-information
        graph = mapper.map(reduced_data, cover=cover)

        # Visualize it
        mapper.visualize(graph, path_html="3DTDA_UU.html", title="3D UofU")

        """
        # Configure the Mapper with necessary parameters:
        graph = mapper.fit_transform(reduced_data, 
                                     cover_params={'num_intervals': num_intervals, 'overlap_frac': overlap_frac}, 
                                     clustering_method=DBSCAN(eps=0.5))
        
        #Step 3: Visualize the Mapper graph
        plt.figure(figsize=(10, 10))  # Set the figure size for the plot.
        nx.draw_networkx(graph, node_size=30, with_labels=False)  # Draw the Mapper graph.
        plt.savefig(output_file)  # Save the graph to the specified output file.
        print(f"Mapper graph saved as {output_file}")  # Inform the user about the saved file.
        """
    except Exception as e:
        print(f"An error occurred: {e}")  # Print any error that occurs.
        raise  # Raise the exception to indicate an error.
        
        

def construct_mapper_graph_2D(data_array, num_intervals, overlap_frac):

    """
    Constructs a Mapper graph for the given data using dimension reduction and clustering.
    
    Parameters:
    - data_array: numpy array of data points.
    - num_intervals: Number of intervals to divide the range of the filter function.
    - overlap_frac: Fractional overlap between consecutive intervals.
    """

    try:
        if data_array.size == 0:
            raise ValueError("Input data array is empty.")
        
        # Initialize
        mapper = km.KeplerMapper(verbose=1)
        
        # Step 1: Dimension reduction using PCA and DB scan
        pca = PCA(n_components=4)  # Reduce the data to 2 principal components.
        reduced_data = pca.fit_transform(data_array)  # Apply PCA on the input data.
        
        """
        What are the 3 columns used for dimensionality? We can use the pce_contributions to determine how much each column is 
        contributing to the clustering. 

        """
        #first convery the numPy to be able to use Pandas 
        feature_names = [f"Column {i}" for i in range(data_array.shape[1])]  # Create generic names if needed
        X = pd.DataFrame(data_array, columns=feature_names)

        # Get contribution of each feature
        pca_contributions = pd.DataFrame(pca.components_, columns=X.columns, index=[f"PC{i+1}" for i in range(pca.n_components_)])
    
        # Show the top contributing features for each principal component
        print(pca_contributions.abs().idxmax(axis=1))
        
        """
        back to the DB scan 
        """
        
        dbscan = DBSCAN(eps=2, min_samples=15) #epsilon is distance between points, min_samples is the minimum number of points required to form a dense region 
        #(i.e., a cluster). A point is considered "core" if it has at least min_samples points (including itself) within its eps-radius.
        clusters = dbscan.fit_predict(reduced_data)

        
        # Step 2: Apply Mapper algorithm
        #mapper = Mapper()  # Initialize the Mapper object from scikit-tda.
        #projected_data = mapper.fit_transform(data_array, projection=[0,1]) # X-Y axis
        
        # Create a cover with 10 elements
        #cover = km.Cover(n_cubes=10)
        cover = km.Cover(n_cubes=num_intervals, perc_overlap=overlap_frac)


        # Create dictionary called 'graph' with nodes, edges and meta-information
        graph = mapper.map(reduced_data, cover=cover)

        # Visualize it
        mapper.visualize(graph, path_html="2DTDA_UU.html", title="2D_UU")

    except Exception as e:
        print(f"An error occurred: {e}")  # Print any error that occurs.
        raise  # Raise the exception to indicate an error.        
"""

# Import the class
import kmapper as km

# Some sample data
from sklearn import datasets
data, labels = datasets.make_circles(n_samples=5000, noise=0.03, factor=0.3)

# Initialize
mapper = km.KeplerMapper(verbose=1)

# Fit to and transform the data
projected_data = mapper.fit_transform(data, projection=[0,1]) # X-Y axis

# Create a cover with 10 elements
cover = km.Cover(n_cubes=10)

# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(projected_data, data, cover=cover)

# Visualize it
mapper.visualize(graph, path_html="make_circles_keplermapper_output.html", title="make_circles(n_samples=5000, noise=0.03, factor=0.3)")


"""




"""
end of script
"""