import json
import importlib

# import graph_aligner
# import graph_distancer
import os
import shutil
import warnings

from graph_aligner import GraphAligner
from graph_distancer import GraphDistancer
from utility import DFKGUtility


class HalluEvaluator:
    def __init__(self, graph_pathA, graph_pathB, matched_pairs_threshold=0.8):
        self.graph_pathA = graph_pathA
        self.graph_pathB = graph_pathB
        self.matched_pairs_threshold = matched_pairs_threshold
        self.graphAligner = GraphAligner(
            self.graph_pathA, self.graph_pathB, self.matched_pairs_threshold
        )
        self.graph1, self.graph2 = self.graphAligner.align_graphs(
            shown_old_ids=False, save_to_file=True
        )
        self.graphDistancer = GraphDistancer(
            graphxA=self.graph1,
            graphxB=self.graph2,
        )
        self.graphx1 = self.graphDistancer.graphx1
        self.graphx2 = self.graphDistancer.graphx2

    def calculate_graph_structural_hullucination(self):
        cost, edit_operations, num_edit_operations = (
            self.graphDistancer.calculate_graph_edit_distance_details()
        )

        match_strings = [
            "num_insert_edge",
            "num_substitute_edge",
            "num_insert_node",
            "num_substitute_node",
        ]

        insert_substitue_ops = sum(
            value for key, value in num_edit_operations.items() if key in match_strings
        )

        num_nodes_edges = len(self.graphx1) + len(self.graphx1.edges())

        return (
            insert_substitue_ops,
            num_nodes_edges,
            100 * insert_substitue_ops / num_nodes_edges,
            edit_operations,
        )

    # Example input
    # graph_keys_inserted: {'1': {'phone', 'ip'}, '2': {'email'}}
    # graph_keys_deleted: {'1': {'birth', 'age'}, '2': {'age'}}
    # graph_keys_kept_same: {'1': {'id', 'name', 'description', 'adddress'}, '2': {'description', 'birth', 'id', 'name', 'address'}}
    def calculate_graph_key_hullucination(self):
        (
            graph_keys_inserted,
            graph_keys_deleted,
            graph_keys_kept_same,
            node_list1,
            node_list2,
        ) = self.graphDistancer.calculate_graph_key_distances()

        number_graph_keys_inserted = sum(
            len(value) for value in graph_keys_inserted.values()
        )

        number_graph_keys_kept_same = sum(
            len(value) for value in graph_keys_kept_same.values()
        )

        key_hullucination = (
            100
            * number_graph_keys_inserted
            / (number_graph_keys_kept_same + number_graph_keys_inserted)
        )

        return (
            number_graph_keys_inserted,
            number_graph_keys_kept_same + number_graph_keys_inserted,
            key_hullucination,
            graph_keys_inserted,
            graph_keys_deleted,
            graph_keys_kept_same,
        )

    def calculate_graph_value_hallucination(self, threshold=0.6):
        # Iput example:
        # {
        # '1': [('name', 0.99999994), ('age', 1.0000001), ('id', 1.0000001), ('birth', 0.32172716), ('description', 1.0000002), ('adddress', 1.0)],
        # '2': [('name', 1.0), ('age', 0.6936892), ('id', 1.0000001), ('birth', 0.5463266), ('description', 1.0000001)]
        # }
        similarities_dict = (
            self.graphDistancer.calculate_value_similarities_from_aligned_nodes()
        )

        count = 0
        total = 0

        for key, value_list in similarities_dict.items():
            for _, number in value_list:
                total += 1
                if number < threshold:
                    count += 1

        return count, total, count / total * 100, similarities_dict

    def summary(self, threshold=0.8):
        (
            num_insert_substitute_ops,
            num_nodes_edges,
            graph_hullucination,
            edit_operations,
        ) = self.calculate_graph_structural_hullucination()

        (
            num_graph_keys_inserted,
            num_graph_keys_kept_same_and_inserted,
            key_hullucination,
            graph_keys_inserted,
            graph_keys_deleted,
            graph_keys_kept_same,
        ) = self.calculate_graph_key_hullucination()

        (
            num_value_below_threshold,
            total_value_count,
            value_hullucination,
            similarities_dict,
        ) = self.calculate_graph_value_hallucination(threshold)

        return (
            num_insert_substitute_ops,
            num_nodes_edges,
            graph_hullucination,
            num_graph_keys_inserted,
            num_graph_keys_kept_same_and_inserted,
            key_hullucination,
            num_value_below_threshold,
            total_value_count,
            value_hullucination,
        )

    @staticmethod
    def test_calculate_value_hallucination():
        print("============Testing test_calcuate_value_hallucination")
        graph_path1 = (
            "testcase/value_distance/01_a.json"  # Path to your graph JSON file
        )
        graph_path2 = (
            "testcase/value_distance/01_b.json"  # Path to your graph JSON file
        )

        halluEvaluator = HalluEvaluator(
            graph_pathA=graph_path1, graph_pathB=graph_path2
        )

        threshold = 0.8
        below_threshold, total_count, percentage, similarities_dict = (
            halluEvaluator.calculate_graph_value_hallucination(threshold)
        )

        print(f"Numbers below {threshold}: {below_threshold}")
        print(f"Total numbers: {total_count}")
        print(f"Percentage below threshold: {percentage:.2f}%")
        print(f"similarities_dict: {similarities_dict}")

    @staticmethod
    def test_calculate_graph_key_hullucination():
        print("============Testing test_calculate_graph_key_hullucination")
        graph_path1 = "testcase/key_distance/03_a.json"  # Path to your graph JSON file
        graph_path2 = "testcase/key_distance/03_b.json"  # Path to your graph JSON file

        halluEvaluator = HalluEvaluator(
            graph_pathA=graph_path1, graph_pathB=graph_path2
        )

        (
            num_graph_keys_inserted,
            num_graph_keys_kept_same,
            key_hullucination,
            graph_keys_inserted,
            graph_keys_deleted,
            graph_keys_kept_same,
        ) = halluEvaluator.calculate_graph_key_hullucination()
        print(f"num_graph_keys_inserted: {num_graph_keys_inserted}")
        print(f"num_graph_keys_kept_same: {num_graph_keys_kept_same}")
        print(
            f"key_hullucination: num_graph_keys_inserted / num_graph_keys_kept_same:{key_hullucination}"
        )
        print(f"graph_keys_inserted: {graph_keys_inserted}")
        print(f"graph_keys_kept_same: {graph_keys_kept_same}")

    @staticmethod
    def test_calculate_graph_key_hullucination_2():
        print("============Testing test_calculate_graph_key_hullucination_2")
        graph_path1 = "testcase/key_distance/07_a.json"  # Path to your graph JSON file
        graph_path2 = "testcase/key_distance/07_b.json"  # Path to your graph JSON file

        halluEvaluator = HalluEvaluator(
            graph_pathA=graph_path1, graph_pathB=graph_path2
        )

        (
            num_graph_keys_inserted,
            num_graph_keys_kept_same,
            key_hullucination,
            graph_keys_inserted,
            graph_keys_deleted,
            graph_keys_kept_same,
        ) = halluEvaluator.calculate_graph_key_hullucination()

        print(f"num_graph_keys_inserted: {num_graph_keys_inserted}")
        print(f"num_graph_keys_kept_same: {num_graph_keys_kept_same}")
        print(
            f"key_hullucination: num_graph_keys_inserted / num_graph_keys_kept_same:{key_hullucination}"
        )
        print(f"graph_keys_inserted: {graph_keys_inserted}")
        print(f"graph_keys_kept_same: {graph_keys_kept_same}")

    @staticmethod
    def test_calculate_graph_structural_hullucination():
        print("============Testing test_calculate_graph_edit_distance_details")
        graph_path1 = "testcase/node_distance/07_a.json"  # Path to your graph JSON file
        graph_path2 = "testcase/node_distance/07_b.json"  # Path to your graph JSON file

        halluEvaluator = HalluEvaluator(
            graph_pathA=graph_path1, graph_pathB=graph_path2
        )

        (
            num_insert_substitute_ops,
            num_nodes_edges,
            graph_hullucination,
            edit_operations,
        ) = halluEvaluator.calculate_graph_structural_hullucination()
        print(f"num_insert_substitute_ops: {num_insert_substitute_ops}")
        print(f"num_nodes_edges: {num_nodes_edges}")
        print(f"graph_hullucination: {graph_hullucination}")
        print(f"edit_operations: {edit_operations}")

        halluEvaluator.graphDistancer.visualize_graphs()
        halluEvaluator.graphDistancer.print_gx(halluEvaluator.graphDistancer.graphx2)

    @staticmethod
    def test_summary():
        print("============Testing sumamry")
        graph_path1 = "testcase/node_distance/09_a.json"  # Path to your graph JSON file
        graph_path2 = "testcase/node_distance/09_b.json"  # Path to your graph JSON file

        halluEvaluator = HalluEvaluator(
            graph_pathA=graph_path1, graph_pathB=graph_path2
        )

        print(halluEvaluator.summary(threshold=0.85))


if __name__ == "__main__":
    # Usage example:

    # HalluEvaluator.test_calculate_value_hallucination()
    # HalluEvaluator.test_calculate_graph_key_hullucination()
    # HalluEvaluator.test_calculate_graph_key_hullucination_2()
    HalluEvaluator.test_calculate_graph_structural_hullucination()
    # HalluEvaluator.test_summary()
