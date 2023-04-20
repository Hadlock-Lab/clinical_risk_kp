import csv
import os
import json


def load_data(data_folder):
    nodes_file_path = os.path.join(data_folder, "ehr_risk_kg_nodes_2021_05_07.tsv")
    edges_file_path = os.path.join(data_folder, "ehr_risk_kg_edges_2021_05_07.tsv")
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
            if current_xref != 'NA':
                subject.update({current_xref.split(':')[0]: current_xref})
            
            # Specify properties for object
            object_ = {
                "id": line[2],
                line[2].split(':')[0]: line[2],
                "name": id_name_mapping[line[2]],
                "type": id_type_mapping[line[2]]
            }
            current_xref = id_xref_mapping[line[2]]
            if current_xref != 'NA':
                object_.update({current_xref.split(':')[0]: current_xref})
            
            # Specify properties for predicate
            predicate = {
                "type": line[1].split(':')[-1] if line[1].startswith("biolink:") else line[1],
                "relation": line[3],
                "provided_by": line[4],
                "provided_date": line[5],
                "provenance": "https://github.com/NCATSTranslator/Translator-All/wiki/EHR-Risk-KP",
                "category": line[6].split(':')[-1] if line[6].startswith("biolink:") else line[6],
                "classifier": line[7],
                "auc_roc": int(float(line[8])*1e4)/1e4
            }

            # Specify conditional properties for predicate (if not 'NA')
            current_feature_imp = line[9]
            if current_feature_imp != 'NA':
                predicate.update({"feature_importance": int(float(line[9]) * 1e4) / 1e4})
            current_coef = line[10]
            if current_coef != 'NA':
                predicate.update({"feature_coefficient": int(float(line[10])*1e4)/1e4})

            # Other properties to include
            predicate.update({"positive_cohort_order_of_magnitude": int(line[11])})
            predicate.update({"negative_cohort_order_of_magnitude": int(line[12])})
                        
            # Yield subject, predicate, and object properties
            _id_list = [line[0], line[1], line[2], line[7], str(line[8])[:6], str(line[9])[:6], str(line[10])[:6]]
            yield {
                "_id": '-'.join(_id_list),
                "subject": subject,
                "predicate": predicate,
                "object": object_
            }
