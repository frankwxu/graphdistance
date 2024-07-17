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
from utility import DFKGUtility


class GraphAligner:
    def __init__(self, graph1_path, graph2_path, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.matched_pairs = {}
        self.new_id_start_graphA = 10000
        self.new_id_start_graphB = 20000
        self.graph1 = self.rename_ids(
            DFKGUtility.load_json(graph1_path), graph_type="A"
        )
        self.graph2 = self.rename_ids(
            DFKGUtility.load_json(graph2_path), graph_type="B"
        )

    # rename add IDs with new ids starting from 10000 (graphA) or 20000 (graph2)
    def rename_ids(self, json_obj, graph_type="A"):
        new_id = (
            self.new_id_start_graphA if graph_type == "A" else self.new_id_start_graphB
        )  # we relabel all IDs with larger numbers
        # Create a mapping of old IDs to new IDs
        id_mapping = {}

        for item in json_obj:
            old_id = item["id"]
            id_mapping[old_id] = str(new_id)
            new_id += 1

        # Update IDs and references
        for item in json_obj:
            item["id"] = id_mapping[item["id"]]

            if "source_ref" in item:
                try:
                    item["source_ref"] = id_mapping[item["source_ref"]]
                except KeyError:
                    raise KeyError(f"source_ref ID doesn't exist: {item['source_ref']}")

            if "target_ref" in item:
                try:
                    item["target_ref"] = id_mapping[item["target_ref"]]
                except KeyError:
                    raise KeyError(f"target_ref ID doesn't exist: {item['target_ref']}")

        # Print the updated data
        print(json_obj)
        print(f"id mapped: {id_mapping}")
        return json_obj

    # input is a dict
    def node_or_edge_to_text(self, anFlattenNode_or_EdgeInSTIX):
        text = ". ".join(
            f"{key}: {value}" for key, value in anFlattenNode_or_EdgeInSTIX.items()
        )
        return text

    # input is a graph
    def graph_to_text(self, a_graph):
        a_graph_text = {}
        for node_or_edge in a_graph:
            flattened_node_or_edge = DFKGUtility.flatten_json(node_or_edge)
            # We don't need the id and we will reassign a new ID
            original_id = flattened_node_or_edge.pop("id")
            a_graph_text[original_id] = self.node_or_edge_to_text(
                flattened_node_or_edge
            )

        return a_graph_text

    def generate_element_embeddings(self, a_graph):
        graph_texts = self.graph_to_text(a_graph)
        # Generate embeddings by converting object attributes to a single string
        return {key: self.model.encode(text) for key, text in graph_texts.items()}

    def compute_similarity_matrix(self, file="similarity_matrix.txt"):
        embeddings1 = np.vstack(
            list(self.generate_element_embeddings(self.graph1).values())
        )
        embeddings2 = np.vstack(
            list(self.generate_element_embeddings(self.graph2).values())
        )

        similarity = cosine_similarity(embeddings1, embeddings2)

        with open(file, "w") as f:
            for row in similarity:
                f.write(" ".join([str(x) for x in row]) + "\n")

        return similarity

    def match_pairs(
        self,
        file="similarity_matrix.txt",
        threshold=0.8,
        saved_file_path="match_pairs.txt",
    ):
        self.compute_similarity_matrix(
            file="similarity_matrix.txt"
        )  # compute the similarity matrix

        with open(file) as f:
            similarity = [
                [float(x) for x in line.strip().split()] for line in f
            ]  # read the matrix from the file

        similarity = np.array(similarity)  # convert to numpy array

        self.matched_pairs = {}

        rows, cols = similarity.shape

        for i in range(rows):
            for j in range(cols):
                if similarity[i, j] > threshold:
                    # Check if the pair or its reverse is not already in matched_pairs
                    if (
                        i not in self.matched_pairs
                        and j not in self.matched_pairs.values()
                    ):
                        self.matched_pairs[i] = j

        # for i, row in enumerate(similarity):
        #     max_score = max(row)
        #     if max_score > threshold:
        #         j = row.index(max_score)
        #         # Check if the pair or its reverse is not already in matched_pairs
        #         if i not in self.matched_pairs and j not in self.matched_pairs.values():
        #             self.matched_pairs[i] = j

        print(f"Matched Pairs: {self.matched_pairs}")
        # Write the dictionary to the file
        with open(saved_file_path, "w") as file:
            json.dump(self.matched_pairs, file, indent=4)

            print(f"Dictionary has been saved to {saved_file_path}")

    # replace id with matched id in a graph
    def replace_id(
        self, a_graph, id_name, matched_pairs, shown_old_ids=False, graph_type="A"
    ):
        # each graph (graphA or graphB) has a differet starting id for each node and edge
        id_start = (
            self.new_id_start_graphA if graph_type == "A" else self.new_id_start_graphB
        )
        for obj in a_graph:
            # make sure we found ID, source_ref, target_ref
            if id_name in obj.keys():
                key = int(obj[id_name]) - id_start
                if key in matched_pairs.keys():
                    if shown_old_ids:
                        obj["old_" + id_name] = key
                    obj[id_name] = str(matched_pairs[key])

                else:
                    if shown_old_ids:
                        obj["old_" + id_name] = obj[id_name]
                    obj[id_name] = obj[id_name]

    def align_graph(self, a_graph, matched_pairs, shown_old_ids=False, graph_type="A"):
        # Update a graph with new IDs
        self.replace_id(a_graph, "id", matched_pairs, shown_old_ids, graph_type)
        self.replace_id(a_graph, "source_ref", matched_pairs, shown_old_ids, graph_type)
        self.replace_id(a_graph, "target_ref", matched_pairs, shown_old_ids, graph_type)

        return a_graph

    def align_graphs(self, shown_old_ids=False, save_to_file=True):
        self.match_pairs(file="similarity_matrix.txt", threshold=0.8)

        self.graph1 = self.align_graph(
            self.graph1, self.matched_pairs, shown_old_ids, graph_type="A"
        )
        self.graph2 = self.align_graph(
            self.graph2, self.matched_pairs, shown_old_ids, graph_type="B"
        )

        if save_to_file:
            with open("graphA.json", "w") as f:
                json.dump(self.graph1, f)
            with open("graphB.json", "w") as f:
                json.dump(self.graph2, f)

        return self.graph1, self.graph2

    @staticmethod
    def test_align_graphs():
        print("============Testing align_graphs")
        graph1_path = "testcase/test/03_a.json"  # Path to your first graph JSON file
        graph2_path = "testcase/test/03_b.json"  # Path to your second graph JSON file

        aligner = GraphAligner(graph1_path, graph2_path)
        g1, g2 = aligner.align_graphs(shown_old_ids=False, save_to_file=True)

        print("g1 and g2 will be saved to graphA.json and graphB.json")
        print(g1)
        print(g2)

    @staticmethod
    def test_rename_ids():
        print("============Testing rename_ids")
        graph_path1 = "testcase/node_distance/06_a.json"  # Path to your graph JSON file
        graph_path2 = "testcase/node_distance/06_b.json"  # Path to your graph JSON file
        aligner = GraphAligner(graph_path1, graph_path2)
        aligner.rename_ids(DFKGUtility.load_json(graph_path1), graph_type="A")


if __name__ == "__main__":
    # Usage example:
    # GraphAligner.test_align_graphs()
    GraphAligner.test_rename_ids()
