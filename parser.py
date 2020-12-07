import csv
import os
import json


def load_data(data_folder):
    edges_file_path = os.path.join(data_folder, "clinical_risk_kg_edges.tsv")
    nodes_file_path = os.path.join(data_folder, "clinical_risk_kg_nodes.tsv")
    nodes_f = open(nodes_file_path)
    edges_f = open(edges_file_path)
    nodes_data = csv.reader(nodes_f, delimiter="\t")
    edges_data = csv.reader(edges_f, delimiter="\t")
    next(nodes_data)
    id_type_mapping = {}
    for line in nodes_data:
        id_type_mapping[line[0]] = line[2].split(':')[-1] if line[2].startswith("biolink:") else line[2]
    next(edges_data)
    for line in edges_data:
        if line[0] and line[1] and line[0].split(':')[0] and line[2].split(':')[0]:
            yield {
                "_id": '-'.join([line[0], line[4], line[5], line[7], str(line[8])]),
                "subject": {
                    "id": line[0],
                    line[0].split(':')[0]: line[0],
                    "name": line[4],
                    "type": id_type_mapping[line[0]]
                },
                "association": {
                    "edge_label": line[1].split(':')[-1],
                    "relation": line[3],
                    "uses_classifier": line[-3],
                    "has_feature_importance": float(line[8]),
                    "has_auc_roc": '' if (line[9] == 'NA') else float(line[9])
                },
                "object": {
                    "id": line[2],
                    line[2].split(':')[0]: line[2],
                    "name": line[5],
                    "type": id_type_mapping[line[2]]
                }
            }
