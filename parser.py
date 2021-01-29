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
    id_name_mapping = {}
    id_type_mapping = {}
    id_xref_mapping = {}
    for line in nodes_data:
        id_name_mapping[line[0]] = line[1]
        id_type_mapping[line[0]] = line[2].split(':')[-1] if line[2].startswith("biolink:") else line[2]
        id_xref_mapping[line[0]] = line[3]
    next(edges_data)
    for line in edges_data:
        if line[0] and line[1] and line[0].split(':')[0] and line[2].split(':')[0]:
            # Specify properties for subject
            subject = {
                "id": line[0],
                line[0].split(':')[0]: line[0],
                "name": id_name_mapping[line[0]],
                "type": id_type_mapping[line[0]]
            }
            current_xref = id_xref_mapping[line[0]]
            if (current_xref != 'None'):
                subject.update({current_xref.split(':')[0]: current_xref})
            
            # Specify properties for object
            object_ = {
                "id": line[2],
                line[2].split(':')[0]: line[2],
                "name": id_name_mapping[line[2]],
                "type": id_type_mapping[line[2]]
            }
            current_xref = id_xref_mapping[line[2]]
            if (current_xref != 'None'):
                object_.update({current_xref.split(':')[0]: current_xref})
            
            # Specify properties for predicate
            predicate = {
                "type": line[1].split(':')[-1] if line[1].startswith("biolink:") else line[1],
                "relation": line[3],
                "provided_by": line[4],
                "category": line[5].split(':')[-1] if line[5].startswith("biolink:") else line[5],
                "classifier": line[6],
                "auc_roc": int(float(line[7])*1e4)/1e4,
                "feature_importance": int(float(line[8])*1e4)/1e4
            }
            current_coef = line[9]
            if (current_coef != 'NA'):
                predicate.update({"feature_coefficient": int(float(line[9])*1e4)/1e4})
                        
            # Yield subject, predicate, and object properties
            yield {
                "_id": '-'.join([line[0], line[1], line[2], line[6], str(line[7])[:6], str(line[8])[:6], str(line[9])[:6]]),
                "subject": subject,
                "predicate": predicate,
                "object": object_
            }
