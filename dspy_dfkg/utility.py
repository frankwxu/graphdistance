import json
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import os
import string
from collections import defaultdict
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import torch


class DFKGUtility:
    @staticmethod
    # All nested attributes within one node/edge are flatten
    # a: {b: v1, c: v2} is flatten to a dictionary {a.b: v1, a.c: v2}
    def flatten_json(anNode_or_EdgeInSTIX):
        out = {}

        def flatten(x, name=""):
            if isinstance(x, dict):
                for a in x:
                    flatten(x[a], name + a + ".")
            elif isinstance(x, list):
                for i, a in enumerate(x):
                    flatten(a, name + str(i) + ".")
            else:
                out[name[:-1]] = x

        flatten(anNode_or_EdgeInSTIX)
        return out

    @staticmethod
    def load_json(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON from file - {file_path}")
            print(f"Error details: {e}")
        return None

    @staticmethod
    def display_flatten_graph(graph_json):
        for ele in graph_json:
            print(DFKGUtility.flatten_json(ele))

    @staticmethod
    def test():
        # Usage example:
        graph_path = "testcase/test/01_a.json"  # Path to your graph JSON file

        graph_json = DFKGUtility.load_json(graph_path)
        print(f"loaded graph in json  result: {graph_json}")
        print("=============")
        print(f"flatted graph")
        DFKGUtility.display_flatten_graph(graph_json)


if __name__ == "__main__":
    DFKGUtility.test()
