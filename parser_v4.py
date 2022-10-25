#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!/usr/bin/env python
# coding: utf-8
# In[13]
# Import Packages
import numpy as np
import pandas as pd
import sys, os
import json
import csv
def load_tsv_data(data_folder):
    """
    This function takes two parameters: edge_filepath and the node_filepath.
    It will return a json formats that has the key of id, subject, predicate, object as the main keys. 
    """
    node_filepath = os.path.join(data_folder, "ehr_risk_nodes_data_2022_09_20.csv")
    edge_filepath = os.path.join(data_folder, "ehr_risk_edges_data_2022_09_20.csv")
    edges_data = pd.read_csv(edge_filepath, sep = ',')
    nodes_data = pd.read_csv(node_filepath, sep = ',')
    cut_off = None
    # generate the nodes atributes
    node_name_mapping = {}
    node_type_mapping = {}
    node_xref_mapping = {}
    for index, row in nodes_data.iterrows():
        node_name_mapping[row["id"]] = row["name"]
        node_type_mapping[row["id"]] = row["category"].split(':')[1] if str(row["category"]).startswith("biolink:") else row["category"]
        node_xref_mapping[row["id"]] = row["xref"] 
    # generate the edges attributes
    for index, row in edges_data.iterrows():
        if row['pvalue']<=0.05:
            if row["subject"] and row["predicate"] and row["subject"].split(':')[0] and row["object"].split(':')[0]:
                # Specify properties for subject
                subject = {
                    row["subject"].split(':')[0]: row["subject"],
                    "id": row["subject"],
                    "name": node_name_mapping[row["subject"]],
                    "type": node_type_mapping[row["subject"]] if pd.notnull(node_type_mapping[row["subject"]]) else "Unknown/Missing"
                }
    #             current_xref = node_xref_mapping[row["subject"]]
                if ~np.isnan(node_xref_mapping[row["subject"]]):
                    subject[node_xref_mapping[row["subject"]].split(':')[0]] =  node_xref_mapping[row["subject"]]
                # Specify properties for object
                objects = {
                    row["object"].split(':')[0]: row["object"],
                    "id": row["object"],
                    "name": node_name_mapping[row["object"]],
                    "type": node_type_mapping[row["object"]] if pd.notnull(node_type_mapping[row["object"]]) else "Unknown/Missing"
                }
                object_aspect = {row["object_aspect"].split(':')[0]: row["object_aspect"].split(':')[-1] if row["object_aspect"].startswith("biolink:") else row["object_aspect"]}
                object_direction = {row["object_direction"].split(':')[0]:row["object_direction"].split(':')[-1] if row["object_direction"].startswith("biolink:") else row["object_direction"]}
                if ~np.isnan(node_xref_mapping[row["object"]]):
                    objects[node_xref_mapping[row["object"]].split(':')[0]] =  node_xref_mapping[row["object"]]
                # Specify properties for predicate
                association = {
                    "predicate": row["predicate"].split(':')[-1] if row["predicate"].startswith("biolink:") else row["predicate"],
    #                     "original": row["predicate"],
                    "provided_date": row["provided_date"],
                    "provenance": "https://github.com/NCATSTranslator/Translator-All/wiki/EHR-Risk-KP",
    #                     "category": row["category"].split(':')[1] if row["category"].startswith("biolink:") else row["category"],
                    "classifier": row["classifier"],
                    "original_predicate": row["relation"],
                    "auc_roc": float(row["auc_roc"]),
                    "p_values": float(row["pvalue"])
                }
                # conditional properties for predicate (if not Null)
                if ~np.isnan(row["feature_importance"]):
                    association["feature_importance"] =  float(row["feature_importance"])
                if ~np.isnan(row["feature_coefficient"]):
                    association["feature_coefficient"] = float(row["feature_coefficient"])
                    association["odd_ratio"] = np.exp(row["feature_coefficient"])

                # make a unique id from  subject, predicate, and object indentifiers
                if row["classifier"] == "Logistic Regression":
                    unique_id_list = [row["subject"], row["predicate"],row["object_aspect"],row["object_direction"], row["object"], row["classifier"], str(row["auc_roc"]), str(row["feature_coefficient"])]
                else:
                    unique_id_list = [row["subject"], row["object_direction"],row["object_aspect"],row["object_direction"], row["object"], row["classifier"], str(row["auc_roc"]),str(row["feature_importance"])]
                json = {
                    "_id":'-'.join(unique_id_list),
                    "subject": subject,
                    "association": association,
                    "object": objects,
                    "object_aspect": object_aspect,
                    "object_direction":object_direction
                    }
                yield json


# In[9]:


# filepath = "./Datasets/"
# def main():
#     count = 0
#     verbose = True
#     for row in load_tsv_data(filepath):
#         if verbose:
#             print(json.dumps(row, sort_keys= False, indent=2))
#         count += 1
#         if count >= 10:
#            break
# if __name__ == "__main__": 
#     main()      


# In[ ]:




