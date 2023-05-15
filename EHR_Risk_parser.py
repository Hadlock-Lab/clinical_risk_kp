#!/usr/bin/env python3

import pandas as pd
import json
import sys, os
import numpy as np

def parse_ehr_risk(data_folder):

    edges_filename = "ehr_risk_edges_data_2022_06_01.csv"
    nodes_filename = "ehr_risk_nodes_data_2022_06_01.csv"

    nodes_filepath = os.path.join(data_folder, nodes_filename)
    edges_filepath = os.path.join(data_folder, edges_filename)
    nodes_data = pd.read_csv(nodes_filepath, sep = ',')
    edges_data = pd.read_csv(edges_filepath, sep = ',')
    
    # the nodes file has duplicate ids; fix in enclave in future
    nodes_data = nodes_data.drop_duplicates(subset='id', keep="first")

    # biolink category biolink:ChemicalSubstance has been deprecated. Use biolink:ChemicalEntity instead
    nodes_data["category"].mask(nodes_data["category"] == "biolink:ChemicalSubstance", "biolink:ChemicalEntity" , inplace=True )
    
    # we originally provided the # of patients with condition --> log + patient count, and # of patients without condition --> log - patient count
    # get the approximate total number of patients in the study and call it "total_sample_size"
    edges_data["num_patients_with_condition"] = 10**(edges_data['log_positive_patient_count']) # convert log pos patient count to an actual # 
    edges_data["num_patients_without_condition"] = 10**(edges_data['log_negative_patient_count']) # convert log neg patient count to an actual #
    edges_data = edges_data.drop(['log_positive_patient_count', 'log_negative_patient_count'], axis=1)
    edges_data["total_sample_size"] = edges_data["num_patients_with_condition"] + edges_data["num_patients_without_condition"]
    edges_data = edges_data.drop(['num_patients_with_condition', 'num_patients_without_condition'], axis=1)
    edges_data["total_sample_size"] = np.random.poisson(edges_data["total_sample_size"]) # add poisson noise injection 

#     # create confidence interval column by concatenating 'lower_confidence_bound'and 'upper_confidence_bound', then dropping those columns
#     edges_data['log_odds_ratio_95_confidence_interval'] = edges_data.apply(lambda row: [row['lower_confidence_bound'], row['upper_confidence_bound']], axis=1)
#     edges_data = edges_data.drop(['lower_confidence_bound', 'upper_confidence_bound'], axis=1)
    
    #  ----- RE-CONSTRUCT KG FROM NODES AND EDGES FILES ------  #
    # merge the subject names, categories and ids from the nodes csv/table to the edges table
    kg = pd.merge(edges_data, nodes_data[['id', 'name', 'category']], left_on='subject', right_on = 'id', how="inner")
    kg.rename(columns = {'category_x':'predicate_category',
                         'category_y': 'subject_category',
                         'id': 'subject_id',
                         'name': 'subject_name'}, inplace = True)
    # merge the object names, categories and ids from the nodes csv/table to the edges table
    kg = pd.merge(kg, nodes_data[['id', 'name', 'category']], left_on='object', right_on = 'id', how="inner")
    kg.rename(columns = {'id':'object_id',
                     'category': 'object_category',
                     'name': 'object_name'}, inplace = True)
    #  ----- ------------------------------------------ ------  #
    
    # ensure there are no duplicates
    kg = kg.drop_duplicates(['subject', 'object', 'auc_roc', 'p_value', 'feature_coefficient'], keep='first')
    
    # some of the subjects/objects contain the string literal "NONE" (specific culprit is COVID Negative or something) Should look into this in future 
    kg = kg[~kg["subject"].str.contains("NONE")==True]  # subject and object are all CURIEs, not names
    kg = kg[~kg["object"].str.contains("NONE")==True]
    kg = kg[~kg["subject"].str.contains("none")==True]
    kg = kg[~kg["object"].str.contains("none")==True]
    kg = kg[~kg["subject"].str.contains("None")==True]
    kg = kg[~kg["object"].str.contains("None")==True]
    
    id_list = [] # use this to check if your document IDs are unique. Collect them and see if they're all unique
    
    # iterate through each row in KG to yield json formatted triple
    for index, row in kg.iterrows(): # comment for testing  
        id_dict = {} # this is the outter dict that holds inner dicts: subject_dict, association_dict, object_dict, and source_dict
        subject_dict = {} # inner dict
        association_dict = {} # inner dict
        object_dict = {} # inner dict
        source_dict = {} # inner dict (provides provenance as per TRAPI 1.4 standards)

        # id generated by concatenating the following: subject_id CURIE, object_id CURIE, AUCROC (removing decimal point) and p-value (removing decimal point), feature coeffcient (removing decimal point), and total sample size
        doc_id = "{}_{}_{}_{}_{}_{}".format(row["subject"],
                                       row["object"],
                                       str(row['auc_roc']),
                                       str(row['p_value']).replace('.', ''),
                                       str(row['feature_coefficient']).replace('.', ''),
                                       str(row["total_sample_size"]))

        id_list.append(doc_id)
        id_dict["_id"] = doc_id
        subject_dict["{}".format(row["subject"].split(':')[0])] = "{}".format(row["subject"].split(':')[1]) # create the subject dict from the rows of the df 
        subject_dict["id"] = row["subject"]
        subject_dict["name"] = row["subject_name"]
        subject_dict["type"] = row["subject_category"]

        association_dict["predicate"] = "{}".format(row["predicate"].split(':')[1]) # create the association dict from the rows of the df. Edge attributes need extra work. The predicate is separated out into qualified predicate by X-BTE annotation, so we don't have to worry about qualifiers here
        association_dict["edge_attributes"] = []

        source_dict["edge_sources"] = []

        association_dict["edge_attributes"].append(
            {
                "attribute_type_id":"biolink:has_supporting_study_result",
                "value":"We train a large collection of multivariable, binary logistic regression models on EHR data for each specific condition/disease/outcome. Features include labs, medications, and phenotypes. Directed edges point from risk factors to specific outcomes (diseases, phenotype, or medication exposure).",
                "attributes": [
                    {
                        "attribute_type_id": "biolink:supporting_study_method_type",
                        "value": "STATO:0000149",
                        "description": "Binomial logistic regression for analysis of dichotomous dependent variable (in this case, for having this particular condition/disease/outcome or not)"
                    },
                    {
                        "attribute_type_id":"biolink:update_date",
                        "value":row["provided_date"]
                    },
                    {
                        "attribute_type_id": "biolink:p_value",
                        "value": float(row["p_value"]),
                        "description": "The p-value represents the probability of observing the estimated coefficient (or more extreme value) under the assumption of the null hypothesis (which assumes that there is no relationship between the independent variable and outcome variable). The p-value associated with each coefficient helps determine whether the relationship between the independent variable and the outcome is statistically significant. A low p-value suggests that the observed relationship between the independent variable and the outcome is unlikely to occur by chance alone, providing evidence against the null hypothesis."
                    },
                    {
                        "attribute_type_id": "STATO:0000209",
                        "value": float(row["auc_roc"]),
                        "description": "The AUROC provides a way to evaluate the model's ability to discriminate between the two classes (the presenece of absence of condition/disease/outcome). Values range between 0-1; the higher the AUROC, the better the model's ability to discriminate between clasess."
                    },
                    {
                        "attribute_type_id": "biolink:log_odds_ratio",
                        "value": float(row['feature_coefficient']),
                        "description": "The logarithm of the odds ratio (log odds ratio), or the ratio of the odds of event Y occurring in an exposed group versus the odds of event Y occurring in a non-exposed group."
                    },
#                     {
#                         "attribute_type_id": "biolink:log_odds_ratio_95_confidence_interval",
#                         "value": row['log_odds_ratio_95_confidence_interval'],
#                         "description": "log_odds_ratio_95_confidence_interval"
#                     },
                    {
                        "attribute_type_id": "biolink:supporting_study_cohort",
                        "value": "age < 18 excluded"
                    },
                    {
                        "attribute_type_id": "biolink:supporting_study_date_range",
                        "value": "2020-2022 (prediction)"
                    },
                    {
                        "attribute_type_id": "biolink:supporting_study_size",
                        "value": int(row["total_sample_size"]),
                        "description": "The total number of patients or participants within a sample population."
                    }
                ]
            }
        )
        association_dict["edge_attributes"].append(
            {
                "attribute_type_id":"biolink:primary_knowledge_source",
                "value":"infores:biothings-multiomics-ehr-risk",
                "value_type_id": "biolink:InformationResource",
                "value_url":  "http://smart-api.info/registry?q=d86a24f6027ffe778f84ba10a7a1861a",
                "description": "The EHR Risk KP is created and maintained by the Multiomics Provider team from the Institute for Systems Biology in Seattle, WA. Through a partnership with Providence/Swedish Health Services and Institute for Systems Biology, we analyze over 26 million EHRs. We use these records to train a large collection of interpretable machine learning models which are integrated into a single large Knowledge Graph, with directed edges pointing from risk factors to specific outcomes (diseases, phenotype, or medication exposure).",
            }
        )
        association_dict["edge_attributes"].append(
            {
                "attribute_type_id":"biolink:supporting_data_source",
                "value":"infores:providence-st-joseph-ehr",
                "value_type_id": "biolink:InformationResource",
                "value_url":  "https://github.com/NCATSTranslator/Translator-All/wiki/EHR-Risk-KP",
                "description": "A partnership with Providence/Swedish Health Services and Institute for Systems Biology allows analysis of 26 million EHRs from patients in seven states in the US, including Alaska, California, Montana, Oregon, Washington, Texas, and New Mexico. Please email data-access@isbscience.org for more information.",
            }
        )

        object_dict["{}".format(row["object"].split(':')[0])] = "{}".format(row["object"].split(':')[1]) # create the object dict from the rows of the df 
        object_dict["id"] = row["object"]
        object_dict["name"] = row["object_name"]
        object_dict["type"] = row["object_category"]

        source_dict["edge_sources"].append(
            {
                "resource_id": "infores:biothings-multiomics-ehr-risk",
                "resource_role": "primary_knowledge_source",
                "upstream_resource_ids": "infores:providence-st-joseph-ehr"
            }
        )

        source_dict["edge_sources"].append(
            {
                "resource_id": "infores:providence-st-joseph-ehr",
                "resource_role": "supporting_data_source"
            }
        )

        id_dict["subject"] = subject_dict # put the subject, association, object, and source dicts into the outer dict called id_dict
        id_dict["association"] = association_dict
        id_dict["object"] = object_dict
        id_dict["source"] = source_dict
            
        # throw error for any rows that are missing any relevant values, such as subject name, subject id/CURIE, subject category, p-value, etc...
        try:
            assert not {x for x in {row["total_sample_size"],
                                    row["subject"],
                                    row["subject_name"],
                                    row["subject_category"],
                                    row["object"],
                                    row["object_name"],
                                    row["object_category"],
                                    row["p_value"],
                                    row["auc_roc"],
                                    row['feature_coefficient']} if x in {None,
                                                                         "NONE",
                                                                         "None",
                                                                         "none",
                                                                         "NA"}}, "Error: All values including subject and object IDs, categories, names, p-value, AUC-ROC, and feature coefficient must be non-null and not contain string literal None or NONE"
#             print(json.dumps(id_dict, indent=2)) # uncomment for testing
#             print(index) # uncomment for testing
            yield id_dict # comment for testing
        except AssertionError as msg:
            print(msg)
    if len(id_list) != len(set(id_list)):
        print("You do not have unique document IDs for each edge in your KG. Either you have duplicate rows/edges, or you simply didn't make a unique identifer (Document ID) for each one.")
    else:
        print("Document IDs appear to be unique")


def main():
	# data_folder = "../../data" # uncomment for testing
	parse_ehr_risk(data_folder) 

if __name__ == "__main__":
	main()

