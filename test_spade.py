from http.client import responses

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
import requests
import lmstudio as lms
#
# joblib.dump(kmeans, "kmeans_model_spade.joblib")
# joblib.dump(scaler, "scaler_spade.joblib")
labels = []
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
            label = "benign"

    labels.append(label)
    return G

with open("log.json", "r", encoding="utf-8") as f:
    log = json.load(f)
graph = build_graph_from_spade(log)

llm = lms.llm("qwen3-30b-a3b")

def graph_to_text(graph: nx.DiGraph) -> str:
    lines = []
    for src, dst, data in graph.edges(data=True):
        src_label = graph.nodes[src].get('label', 'Unknown')
        dst_label = graph.nodes[dst].get('label', 'Unknown')
        op = data.get('label', data.get('operation', 'access'))
        lines.append(f"{src} ({src_label}) --[{op}]--> {dst} ({dst_label})")
    return "\n".join(lines)


def llm_analyze_graph(graph):
    prompt = graph_to_text(graph)
    full_prompt = f"""
    You are a cybersecurity assistant.
    Analyze the following provenance graph of a potential attack:

    {prompt}
    """

    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(full_prompt)
    responses = llm.respond(full_prompt)
    return responses

print(llm_analyze_graph(graph))