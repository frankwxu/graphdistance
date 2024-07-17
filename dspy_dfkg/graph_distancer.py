import json
import random
import os
import string
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from utility import DFKGUtility


class GraphDistancer:

    def __init__(
        self,
        graph_pathA=None,
        graph_pathB=None,
        graphxA=None,
        graphxB=None,
        model_name="all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)
        if graph_pathA is not None and graph_pathB is not None:
            print("Both paths are None! Using aligned graphs.")
            self.graph1 = DFKGUtility.load_json(graph_pathA)
            self.graph2 = DFKGUtility.load_json(graph_pathB)
        else:
            self.graph1 = graphxA
            self.graph2 = graphxB
        self.graphx1 = self.build_networkx_graph(self.graph1)
        self.graphx2 = self.build_networkx_graph(self.graph2)

    def is_edge(self, node_or_edge_in_json):  # node_or_edge_in_json is a dict
        return ("source_ref" in node_or_edge_in_json.keys()) or (
            "target_ref" in node_or_edge_in_json.keys()
        )

    def contains_ref_string(self, node_or_edge_in_json):
        for k in node_or_edge_in_json:
            if k.endswith("_ref") or k.endswith("_refs"):
                return True

    def build_networkx_graph(self, graph_json):
        """
        Build a NetworkX graph from a list of JSON objects.

        Parameters:
        json_list (list): A list of dictionaries, each representing a node with its connections.
                        Each dictionary should have 'id', 'name', and 'connections' keys.

        Returns:
        networkx.Graph: A NetworkX graph representation of the input data.
        """
        # Create a new NetworkX graph
        print("Strat building networkx graph!")
        G = nx.Graph()

        # Add nodes and edges based on the JSON data

        for item in graph_json:
            # Add node
            if not self.is_edge(item):
                G.add_node(
                    item.get("id"),
                    id=item.get("id"),
                )

        for item in graph_json:
            # Add node
            if self.is_edge(item):
                source = item.get("source_ref")
                target = item.get("target_ref")
                # print(f"{source}-->{target}")

                if source in G.nodes and target in G.nodes and source != target:
                    # We rename the edge ID as concat of source+target ids
                    # or we can use orginal id G.add_edge(source, target, id=item.get("id"))
                    G.add_edge(
                        source, target, id=source + "_" + item.get("id") + "_" + target
                    )
            else:
                for k in item.keys():
                    if k.endswith("_ref"):
                        G.add_edge(
                            item["id"],
                            item[k],
                            id=item["id"] + "_" + k + "_" + item.get(k),
                        )
                    if k.endswith("_refs"):
                        for refID in item.get(k):
                            G.add_edge(
                                item["id"],
                                refID,
                                id=item["id"] + "_" + k + "_" + refID,
                            )

        return G

    def print_gx(self, graphx):
        print("Nodes:", list(graphx.nodes()))
        print("Edges:")

        for u, v in graphx.edges():
            print(f"  ({u}, {v})")

    def visualize_graphs(self):
        plt.figure(figsize=(20, 10))

        # Graph 1
        plt.subplot(121)
        pos1 = nx.spring_layout(self.graphx1)
        nx.draw(
            self.graphx1,
            pos1,
            with_labels=True,
            node_color="lightblue",
            node_size=2500,
            font_size=20,
        )
        # Draw edge labels
        edge_labels1 = nx.get_edge_attributes(self.graphx1, "id")
        nx.draw_networkx_edge_labels(
            self.graphx1, pos1, edge_labels=edge_labels1, font_size=20
        )
        plt.title("Graph 1")

        # Graph 2
        plt.subplot(122)
        pos2 = nx.spring_layout(self.graphx2)
        nx.draw(
            self.graphx2,
            pos2,
            with_labels=True,
            node_color="lightgreen",
            node_size=2500,
            font_size=20,
        )
        # Draw edge labels
        edge_labels2 = nx.get_edge_attributes(self.graphx2, "id")
        nx.draw_networkx_edge_labels(
            self.graphx2, pos2, edge_labels=edge_labels2, font_size=20
        )
        plt.title("Graph 2")

        plt.tight_layout()
        plt.show()

    def calculate_graph_edit_distance(self):
        """
        Only produce results, no edit detials
        """

        def node_match(n1, n2):
            return n1.get("id") == n2.get("id")

        def edge_match(e1, e2):
            return e1.get("id") == e2.get("id")

        # Custom cost functions to display all operations
        def node_subst_cost(n1, n2):
            cost = 0 if node_match(n1, n2) else 1
            print(f"Node substitution: {n1} -> {n2}, Cost: {cost}")
            return cost

        def node_del_cost(n):
            print(f"Node deletion: {n}, Cost: 1")
            return 1

        def node_ins_cost(n):
            print(f"Node insertion: {n}, Cost: 1")
            return 1

        def edge_subst_cost(e1, e2):
            cost = 0 if edge_match(e1, e2) else 1
            print(f"Edge substitution: {e1} -> {e2}, Cost: {cost}")
            return cost

        def edge_del_cost(e):
            print(f"Edge deletion: {e}, Cost: 1")
            return 1

        def edge_ins_cost(e):
            print(f"Edge insertion: {e}, Cost: 1")
            return 1

        ged = nx.graph_edit_distance(
            self.graphx1,
            self.graphx2,
            node_match=node_match,
            edge_match=edge_match,
            node_subst_cost=node_subst_cost,
            node_del_cost=node_del_cost,
            node_ins_cost=node_ins_cost,
            edge_subst_cost=edge_subst_cost,
            edge_del_cost=edge_del_cost,
            edge_ins_cost=edge_ins_cost,
        )

        return ged

    def calculate_graph_edit_distance_details(self):
        """
        Calculate the graph edit distance between two undirected graphs and list each edit operation.

        Parameters:
        graphx1, graphx2 (networkx.Graph): The two graphs to compare

        Returns:
        float: The graph edit distance
        list: List of edit operations
        """
        # Calculate the edit paths and distance
        edit_paths = list(nx.optimize_edit_paths(self.graphx1, self.graphx2))

        # The first item in the list is the best edit path
        best_path = edit_paths[0]
        vertex_path, edge_path, cost = best_path

        edit_operations = defaultdict(list)
        num_edit_operations = defaultdict(int)

        # List vertex edit operations
        for v1, v2 in vertex_path:
            if v1 is None:
                edit_operations["insert_node"].append(v2)
                num_edit_operations["num_insert_node"] += 1
            elif v2 is None:
                edit_operations["delete_node"].append(v1)
                num_edit_operations["num_delete_node"] += 1
            elif v1 != v2:
                edit_operations["substitute_node"].append((v1, v2))
                num_edit_operations["num_substitute_node"] += 1

        # List edge edit operations
        for e1, e2 in edge_path:
            if e1 is None:
                edit_operations["insert_edge"].append(e2)
                num_edit_operations["num_insert_edge"] += 1
            elif e2 is None:
                edit_operations["delete_edge"].append(e1)
                num_edit_operations["num_delete_edge"] += 1
            elif e1 != e2:
                edit_operations["substitute_edge"].append((e1, e2))
                num_edit_operations["num_substitute_edge"] += 1

        # it seems cost doesn't always return correct results
        # use dict(num_edit_operations) instead
        return cost, dict(edit_operations), dict(num_edit_operations)

    def get_aligned_nodes(self):
        id_set1 = set(element.get("id") for element in self.graph1)
        id_set2 = set(element.get("id") for element in self.graph2)
        intersection = id_set1.intersection(id_set2)
        return [
            element for element in self.graph1 if element.get("id") in intersection
        ], [element for element in self.graph2 if element.get("id") in intersection]

    def calculate_key_distances(self, node_or_edge_a, node_or_edge_b):
        g1_keys = set(node_or_edge_a.keys())
        g2_keys = set(node_or_edge_b.keys())

        keys_inserted = g2_keys - g1_keys
        keys_deleted = g1_keys - g2_keys
        keys_kept_same = g1_keys.intersection(g2_keys)

        return node_or_edge_a.get("id"), keys_inserted, keys_deleted, keys_kept_same

    # only calcuate distance in the aligned nodes and edges
    def calculate_graph_key_distances(self, ignore_id=True):
        graph_keys_inserted = {}
        graph_keys_deleted = {}
        graph_keys_kept_same = {}

        node_list1, node_list2 = self.get_aligned_nodes()

        for node1, node2 in zip(node_list1, node_list2):
            id, inserted, deleted, kept_same = self.calculate_key_distances(
                node1, node2
            )
            graph_keys_inserted[id] = inserted
            graph_keys_deleted[id] = deleted
            graph_keys_kept_same[id] = kept_same

        if ignore_id:
            graph_keys_kept_same = {
                key: value - {"id"} for key, value in graph_keys_kept_same.items()
            }

        return (
            graph_keys_inserted,
            graph_keys_deleted,
            graph_keys_kept_same,
            node_list1,
            node_list2,
        )

    # precondition: matched nodes and keys
    # return a dictionary of {key: value} in both node/edge a and node/edge b
    def get_aligned_values_from_aligned_node_or_edge(
        self, node_or_edge_a, node_or_edge_b
    ):
        g1_keys = set(node_or_edge_a.keys())
        g2_keys = set(node_or_edge_b.keys())

        keys_kept_same = g1_keys.intersection(g2_keys)

        result = {
            k: (node_or_edge_a[k], node_or_edge_b[k])
            for k in keys_kept_same
            if k in node_or_edge_a and k in node_or_edge_b
        }

        return result

    # ONE parid nodes
    def calculate_value_similarities(
        self, paried_agligned_keys_values_dict_in_node, skip_id=True
    ):
        similarities = []

        # Iterate over items() instead of keys to get both key and value
        for k, (value1, value2) in paried_agligned_keys_values_dict_in_node.items():
            if skip_id and k == "id":
                continue
            # Convert values to strings
            str_value1 = str(value1)
            str_value2 = str(value2)

            # Encode the values
            embedding1 = self.model.encode([str_value1])
            embedding2 = self.model.encode([str_value2])

            # Compute cosine similarity
            sim_score = cosine_similarity(embedding1, embedding2)[0][0]

            similarities.append((k, sim_score))  # Include key in the result

        return similarities

    def calculate_value_similarities_from_aligned_nodes(self):
        node_list1, node_list2 = self.get_aligned_nodes()
        value_similarity_dict_per_node_edge = {}
        for node1, node2 in zip(node_list1, node_list2):
            # {k: (v1, v2)}
            paried_agligned_keys_values_dict_in_node = (
                self.get_aligned_values_from_aligned_node_or_edge(node1, node2)
            )
            similarity_of_a_node_or_edge = self.calculate_value_similarities(
                paried_agligned_keys_values_dict_in_node
            )
            value_similarity_dict_per_node_edge[node1.get("id")] = (
                similarity_of_a_node_or_edge
            )

        return value_similarity_dict_per_node_edge

    @staticmethod
    def test_get_aligned_values_from_aligned_node_or_edge():
        graph_path1 = "testcase/test/05_a.json"  # Path to your graph JSON file
        graph_path2 = "testcase/test/05_b.json"  # Path to your graph JSON file

        graph_dist = GraphDistancer(graph_pathA=graph_path1, graph_pathB=graph_path2)
        node1 = graph_dist.graph1[0]
        node2 = graph_dist.graph2[0]

        print(node1)

        key_alue_pairs_dict = graph_dist.get_aligned_values_from_aligned_node_or_edge(
            node1, node2
        )

        for k in key_alue_pairs_dict:
            print(key_alue_pairs_dict.get(k))

    @staticmethod
    def test_calculate_value_similarities():
        print("============Testing test_calculate_value_similarities")
        graph_path1 = "testcase/test/05_a.json"  # Path to your graph JSON file
        graph_path2 = "testcase/test/05_b.json"  # Path to your graph JSON file

        graph_dist = GraphDistancer(graph_pathA=graph_path1, graph_pathB=graph_path2)
        node1 = graph_dist.graph1[0]
        node2 = graph_dist.graph2[0]

        print(node1)

        key_alue_pairs_dict = graph_dist.get_aligned_values_from_aligned_node_or_edge(
            node1, node2
        )

        for k, (v1, v2) in key_alue_pairs_dict.items():
            print(key_alue_pairs_dict.get(k))

        similarity = graph_dist.calculate_value_similarities(key_alue_pairs_dict)

        print(similarity)

    @staticmethod
    def test_calculate_value_similarities_from_aligned_nodes():
        print(
            "============Testing test_calculate_value_similarities_from_aligned_nodes"
        )
        graph_path1 = "testcase/test/06_a.json"  # Path to your graph JSON file
        graph_path2 = "testcase/test/06_b.json"  # Path to your graph JSON file

        graph_dist = GraphDistancer(graph_pathA=graph_path1, graph_pathB=graph_path2)

        print(graph_dist.calculate_value_similarities_from_aligned_nodes())

    @staticmethod
    def test_calculate_value_hallucination():
        print("============Testing test_calcuate_value_hallucination")
        graph_path1 = "testcase/test/06_a.json"  # Path to your graph JSON file
        graph_path2 = "testcase/test/06_b.json"  # Path to your graph JSON file

        graph_dist = GraphDistancer(graph_pathA=graph_path1, graph_pathB=graph_path2)

        threshold = 0.6
        below_threshold, total_count, percentage = (
            graph_dist.calculate_graph_value_hallucination(threshold)
        )

        print(f"Numbers below {threshold}: {below_threshold}")
        print(f"Total numbers: {total_count}")
        print(f"Percentage below threshold: {percentage:.2f}%")

    @staticmethod
    def test_calculate_graph_key_distances():
        print("============Testing test_calculate_graph_key_distances")
        graph_path1 = "testcase/key_distance/05_a.json"  # Path to your graph JSON file
        graph_path2 = "testcase/key_distance/05_b.json"  # Path to your graph JSON file

        graph_dist = GraphDistancer(graph_pathA=graph_path1, graph_pathB=graph_path2)

        (
            graph_keys_inserted,
            graph_keys_deleted,
            graph_keys_kept_same,
            node_list1,
            node_list2,
        ) = graph_dist.calculate_graph_key_distances()
        print(f"graph_keys_inserted: {graph_keys_inserted}")
        print(f"graph_keys_deleted: {graph_keys_deleted}")
        print(f"graph_keys_kept_same: {graph_keys_kept_same}")
        print(f"node_list1: {node_list1}")
        print(f"node_list2: {node_list2}")

    @staticmethod
    def test_calculate_graph_key_distances_2():
        print("============Testing test_calculate_graph_key_distances")
        graph_path1 = "testcase/key_distance/07_a.json"  # Path to your graph JSON file
        graph_path2 = "testcase/key_distance/07_b.json"  # Path to your graph JSON file

        graph_dist = GraphDistancer(graph_pathA=graph_path1, graph_pathB=graph_path2)

        (
            graph_keys_inserted,
            graph_keys_deleted,
            graph_keys_kept_same,
            node_list1,
            node_list2,
        ) = graph_dist.calculate_graph_key_distances()
        print(f"graph_keys_inserted: {graph_keys_inserted}")
        print(f"graph_keys_deleted: {graph_keys_deleted}")
        print(f"graph_keys_kept_same: {graph_keys_kept_same}")
        print(f"node_list1: {node_list1}")
        print(f"node_list2: {node_list2}")

    @staticmethod
    def test_calculate_graph_key_hullucination():
        print("============Testing test_calculate_graph_key_hullucination")
        graph_path1 = "testcase/key_distance/06_a.json"  # Path to your graph JSON file
        graph_path2 = "testcase/key_distance/06_b.json"  # Path to your graph JSON file

        graph_dist = GraphDistancer(graph_pathA=graph_path1, graph_pathB=graph_path2)

        (
            graph_keys_inserted,
            graph_keys_deleted,
            graph_keys_kept_same,
            node_list1,
            node_list2,
        ) = graph_dist.calculate_graph_key_distances()
        print(f"graph_keys_inserted: {graph_keys_inserted}")
        print(f"graph_keys_deleted: {graph_keys_deleted}")
        print(f"graph_keys_kept_same: {graph_keys_kept_same}")

        num_graph_keys_inserted, num_graph_keys_kept_same, key_hullucination = (
            graph_dist.calculate_graph_key_hullucination()
        )
        print(f"graph_keys_inserted: {num_graph_keys_inserted}")
        print(f"graph_keys_kept_same: {num_graph_keys_kept_same}")
        print(
            f"key_hullucination: num_graph_keys_inserted / num_graph_keys_kept_same:{key_hullucination}"
        )

    @staticmethod
    def test_calculate_graph_key_hullucination_2():
        print("============Testing test_calculate_graph_key_hullucination_2")
        graph_path1 = "testcase/key_distance/07_a.json"  # Path to your graph JSON file
        graph_path2 = "testcase/key_distance/07_b.json"  # Path to your graph JSON file

        graph_dist = GraphDistancer(graph_pathA=graph_path1, graph_pathB=graph_path2)

        (
            graph_keys_inserted,
            graph_keys_deleted,
            graph_keys_kept_same,
            node_list1,
            node_list2,
        ) = graph_dist.calculate_graph_key_distances()
        print(f"graph_keys_inserted: {graph_keys_inserted}")
        print(f"graph_keys_deleted: {graph_keys_deleted}")
        print(f"graph_keys_kept_same: {graph_keys_kept_same}")

        num_graph_keys_inserted, num_graph_keys_kept_same, key_hullucination = (
            graph_dist.calculate_graph_key_hullucination()
        )
        print(f"graph_keys_inserted: {num_graph_keys_inserted}")
        print(f"graph_keys_kept_same: {num_graph_keys_kept_same}")
        print(
            f"key_hullucination: num_graph_keys_inserted / num_graph_keys_kept_same:{key_hullucination}"
        )

    @staticmethod
    def test_calculate_graph_edit_distance():
        print("============Testing test_calculate_graph_edit_distance")
        graph_path1 = "testcase/node_distance/02_a.json"  # Path to your graph JSON file
        graph_path2 = "testcase/node_distance/02_b.json"  # Path to your graph JSON file

        distancer = GraphDistancer(graph_pathA=graph_path1, graph_pathB=graph_path2)
        print(f"Edit distance: {distancer.calculate_graph_edit_distance()}")
        distancer.visualize_graphs()

    @staticmethod
    def test_calculate_graph_edit_distance_details():
        print("============Testing test_calculate_graph_edit_distance_details")
        graph_path1 = "testcase/node_distance/07_a.json"  # Path to your graph JSON file
        graph_path2 = "testcase/node_distance/07_b.json"  # Path to your graph JSON file

        distancer = GraphDistancer(graph_pathA=graph_path1, graph_pathB=graph_path2)

        cost, edit_op, num_edit_op = distancer.calculate_graph_edit_distance_details()
        print(f"Edit distance: {cost}")
        print(f"Edit distance operation: {edit_op}")
        print(f"Number of edit distance Operation: {num_edit_op}")

        # print(distancer.calculate_graph_edit_distance_details())

        distancer.visualize_graphs()

    @staticmethod
    def test_calculate_graph_structural_hullucination():
        print("============Testing test_calculate_graph_edit_distance_details")
        graph_path1 = "testcase/node_distance/07_a.json"  # Path to your graph JSON file
        graph_path2 = "testcase/node_distance/07_b.json"  # Path to your graph JSON file

        distancer = GraphDistancer(graph_pathA=graph_path1, graph_pathB=graph_path2)

        num_insert_substitute_ops, num_nodes_edges, graph_hullucination = (
            distancer.calculate_graph_structural_hullucination()
        )
        print(f"num_insert_substitute_ops: {num_insert_substitute_ops}")
        print(f"num_nodes_edges: {num_nodes_edges}")
        print(f"graph_hullucination: {graph_hullucination}")

        distancer.visualize_graphs()


if __name__ == "__main__":
    # Usage example:
    # GraphDistancer.test_get_aligned_values_from_aligned_node_or_edge()
    # GraphDistancer.test_calculate_value_similarities()
    # GraphDistancer.test_calculate_value_similarities_from_aligned_nodes()
    # GraphDistancer.test_calculate_value_hallucination()
    # GraphDistancer.test_calculate_graph_key_distances()
    # GraphDistancer.test_calculate_graph_key_distances_2()
    # GraphDistancer.test_calculate_graph_key_hullucination()
    # GraphDistancer.test_calculate_graph_key_hullucination_2()
    # GraphDistancer.test_calculate_graph_edit_distance()
    GraphDistancer.test_calculate_graph_edit_distance_details()
    # GraphDistancer.test_calculate_graph_structural_hullucination()
