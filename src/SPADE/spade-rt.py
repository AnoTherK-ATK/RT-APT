import pandas as pd
import numpy as np
import hashlib
import networkx as nx
from collections import Counter, deque
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
import random as rd
import json
import pexpect
import time

kmeans = joblib.load("kmeans_model.joblib")
scaler = joblib.load("scaler.joblib")

# Load threshold
with open("threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

class FlexSketch:
    def __init__(self, max_bins=100, max_histograms=5):
        self.max_bins = max_bins
        self.max_histograms = max_histograms
        self.histograms = deque()
        self.weights = deque()

    def update(self, label_counter):
        most_common = label_counter.most_common(self.max_bins)
        vector = np.zeros(self.max_bins)
        for i, (_, count) in enumerate(most_common):
            vector[i] = count
        self.histograms.append(vector)
        self.weights.append(1.0)
        if len(self.histograms) > self.max_histograms:
            self.histograms.popleft()
            self.weights.popleft()
        total_weight = sum(self.weights)
        self.weights = deque([w / total_weight for w in self.weights])

    def estimate_vector(self):
        result = np.zeros(self.max_bins)
        for h, w in zip(self.histograms, self.weights):
            result += w * h
        return result

# WL subtree extraction
def wl_subtree_features(graph, k=2):
    node_labels = nx.get_node_attributes(graph, 'label')
    features = {node: [node_labels.get(node, 'N/A')] for node in graph.nodes()}
    current_labels = node_labels.copy()
    for _ in range(k):
        new_labels = {}
        for node in graph.nodes():
            neighbors = sorted(
                [str(current_labels.get(nbr, '')) for nbr in graph.predecessors(node)] +
                [str(current_labels.get(nbr, '')) for nbr in graph.successors(node)]
            )
            combined = str(current_labels.get(node, '')) + "|" + "|".join(neighbors)
            hash_label = hashlib.md5(combined.encode()).hexdigest()
            new_labels[node] = hash_label
            features[node].append(hash_label)
        current_labels = new_labels
    return features

def flexsketch_vector_from_graph(graph: nx.DiGraph, k: int = 2, max_bins: int = 100, max_histograms: int = 5):
    wl_feats = wl_subtree_features(graph, k)
    all_labels = []
    for labels in wl_feats.values():
        all_labels.extend(labels)
    sketch = FlexSketch(max_bins=max_bins, max_histograms=max_histograms)
    sketch.update(Counter(all_labels))
    return sketch.estimate_vector()

def build_graph_from_spade(events):
    G = nx.DiGraph()

    for entry in events:
        if 'id' in entry:  # là một node
            node_id = entry['id']
            node_type = entry['type']
            label = entry.get('annotations', {}).get('subtype', node_type)
            G.add_node(node_id, label=label)
        else:  # là một edge
            src = entry['from']
            dst = entry['to']
            label = entry['annotations'].get("operation", entry["type"])
            G.add_edge(src, dst, label=label)

    return G

def min_distance(x):
    return np.min(np.linalg.norm(kmeans.cluster_centers_ - x, axis=1))

def detect(x):
    if(min_distance(x) < threshold):
        return 0
    else:
        return 1

# Đường dẫn tới file log gốc và nơi lưu file tổng hợp
SOURCE_JSON_PATH = "/home/kda/log.json"

def real_time_spade(cycle_count=3, wait_seconds=30):
    print(f"[+] Starting spade control for {cycle_count} cycles...")
    child = pexpect.spawn("spade control", encoding='utf-8')
    child.expect("->")
    i = 0
    while(True):
        i += 1
        print(f"\n===== Cycle {i} / {cycle_count} =====")

        # Add reporter
        child.sendline("add storage JSON output=/home/kda/log.json")
        child.expect("->")
        print(f"[+] [{i}] Storage added.")

        # Wait
        print(f"[*] [{i}] Waiting {wait_seconds} seconds...")
        time.sleep(wait_seconds)

        # Remove reporter
        child.sendline("remove storage JSON")
        child.expect("->")
        print(f"[+] [{i}] Storage removed.")

        # Append collected log
        with open(SOURCE_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            graph = build_graph_from_spade(data)
            vector = flexsketch_vector_from_graph(graph, k=3, max_bins=100, max_histograms=5)
            vector_scaled = scaler.transform([vector])
            if(detect(vector_scaled[0])):
                print(f"[+] [{i}] attack detected.")
            else:
                print(f"[+] [{i}] benign.")

    child.sendline("exit")
    child.expect(pexpect.EOF)
    print("\n[+] Finished all cycles.")

if __name__ == "__main__":
    real_time_spade()