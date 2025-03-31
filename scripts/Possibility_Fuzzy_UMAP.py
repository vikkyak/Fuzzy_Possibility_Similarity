#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:37:57 2025

@author: user
"""

#!/usr/bin/env python3
# File: similarity_graph.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import sys


def possibility_kernel(xi, xj, alpha):
    xi = xi.reshape(1, -1)
    xj = xj.reshape(1, -1)
    sim = cosine_similarity(xi, xj)[0][0]
    return np.exp(-alpha * (1 - sim))


def create_possibility_similarity_graph(data, n_neighbors=10, alpha=1.0, threshold=0.5):
    num_points = len(data)
    possibility_matrix = np.zeros((num_points, num_points))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    distances, indices = nbrs.kneighbors(data)

    for i in range(num_points):
        for j in indices[i][1:]:
            pij = possibility_kernel(data[i], data[j], alpha)
            pji = possibility_kernel(data[j], data[i], alpha)
            sim = max(min(pij, pji), 0)
            sim = sim if sim >= threshold else 0
            possibility_matrix[i, j] = sim
            possibility_matrix[j, i] = sim

    return possibility_matrix


def fuzzy_similarity(xi, xj, rho_i, sigma_i):
    xi = xi.reshape(1, -1)
    xj = xj.reshape(1, -1)
    similarity = cosine_similarity(xi, xj)[0][0]
    distance = 1 - similarity
    if distance - rho_i <= 0:
        return 1.0
    else:
        return np.exp(-(distance - rho_i) / sigma_i)


def create_fuzzy_similarity_graph(data, n_neighbors=10):
    num_points = len(data)
    fuzzy_matrix = np.zeros((num_points, num_points))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    distances, indices = nbrs.kneighbors(data)

    rhos = np.min(distances[:, 1:], axis=1)
    sigmas = np.zeros(num_points)

    for i in range(num_points):
        target = np.log2(n_neighbors)
        sigma = 1.0
        for _ in range(100):
            psum = np.sum([np.exp(-(d - rhos[i]) / sigma) if d > rhos[i] else 1.0 for d in distances[i][1:]])
            if abs(psum - target) < 1e-3:
                break
            sigma *= psum / target
        sigmas[i] = sigma

    for i in range(num_points):
        for j in indices[i][1:]:
            pij = fuzzy_similarity(data[i], data[j], rhos[i], sigmas[i])
            pji = fuzzy_similarity(data[j], data[i], rhos[j], sigmas[j])
            fuzzy_val = pij + pji - pij * pji
            fuzzy_matrix[i, j] = fuzzy_val
            fuzzy_matrix[j, i] = fuzzy_val

    return fuzzy_matrix


def build_graph_from_matrix(matrix):
    num_points = matrix.shape[0]
    G = nx.Graph()
    for i in range(num_points):
        G.add_node(i)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            weight = matrix[i, j]
            if weight > 0:
                G.add_edge(i, j, weight=weight)
    return G


# def visualize_graph(G, title):
#     pos = nx.spring_layout(G, seed=42)
#     edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
#     plt.figure(figsize=(8, 6))
#     nx.draw(G, pos, with_labels=True, node_color='lightblue',
#             edge_color=edge_weights, width=2, edge_cmap=plt.cm.Greens, node_size=500)
#     plt.title(title)
#     plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Greens), label="Edge Weight")
#     plt.tight_layout()
#     plt.savefig(f"{title.replace(' ', '_').lower()}.png")

def visualize_graph(G, title):
    pos = nx.spring_layout(G, seed=42)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    plt.figure(figsize=(8, 6))
    ax = plt.gca()  # get current axes
    edges = nx.draw(
        G, pos, ax=ax, with_labels=True, node_color='lightblue',
        edge_color=edge_weights, width=2, edge_cmap=plt.cm.Greens, node_size=500
    )
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Greens)
    sm.set_array(edge_weights)
    plt.colorbar(sm, ax=ax, label="Edge Weight")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()




if __name__ == "__main__":
    np.random.seed(42)
    data = np.random.rand(100, 10)

    possibility_matrix = create_possibility_similarity_graph(data, n_neighbors=10, alpha=0.5, threshold=0.3)
    G_poss = build_graph_from_matrix(possibility_matrix)
    visualize_graph(G_poss, "UMAP-Style Possibility Similarity Graph")

    fuzzy_matrix = create_fuzzy_similarity_graph(data, n_neighbors=10)
    G_fuzzy = build_graph_from_matrix(fuzzy_matrix)
    visualize_graph(G_fuzzy, "UMAP-Style Fuzzy Similarity Graph")
