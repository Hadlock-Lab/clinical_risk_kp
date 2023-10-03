# Databricks notebook source
# DBTITLE 1,Load CEDA Tools
# MAGIC %run
# MAGIC "/Users/qi.wei1@providence.org/CEDA-tool-test/CEDA Tools v1.1 test/load_ceda_api"

# COMMAND ----------

# DBTITLE 1,Specify the version number, whether to use today's date or manual input date
## Read the current date as a version string
from pyspark.sql.functions import current_date
version_number = str(spark.range(1).withColumn("date",current_date()).select("date").collect()[0][0]).replace('-', '_')
print("The version number added will be {}".format(version_number))

## Or manually load a version number
## Notice then remember to comment out this following part if you don't want the version number to be overwritten
version_number = "2023_05_31"
print("The manual selected version number added will be {}".format(version_number))

# COMMAND ----------

# DBTITLE 1,If unwanted files being saved, use following code to remove those unwanted files
# Get labels for model results
# results_folder_name = 'all_cond_lr_models_2023_05_31_CI_pValue/'
# lr_models_save_location = "abfss://redap-isb-all@stgredapuserrw.dfs.core.windows.net/rdp_phi_sandbox/hadlock_common/translator/{}".format(results_folder_name)
# result_files = dbutils.fs.ls(lr_models_save_location)

# print(result_files)

## Delete chronic_hepatitis_B_C & HIV
# dbutils.fs.rm("abfss://redap-isb-all@stgredapuserrw.dfs.core.windows.net/rdp_phi_sandbox/hadlock_common/translator/all_cond_lr_models_2023_05_31_CI_pValue/chronic_hepatitis_B_C_lr_model_info.json")
# dbutils.fs.rm("abfss://redap-isb-all@stgredapuserrw.dfs.core.windows.net/rdp_phi_sandbox/hadlock_common/translator/all_cond_lr_models_2023_05_31_CI_pValue/HIV_lr_model_info.json")

# COMMAND ----------

# DBTITLE 1,Load logistic regression results and save to sandbox table (could use multi_processing to speed up as well)
# Get labels for model results
results_folder_name = 'all_cond_lr_models_2023_08_10_CI_pValue/'
lr_models_save_location = "abfss://redap-isb-all@stgredapuserrw.dfs.core.windows.net/rdp_phi_sandbox/hadlock_common/translator/{}".format(results_folder_name)
result_files = dbutils.fs.ls(lr_models_save_location)
## Function to get the outcome lable from the file name
def get_label(file): return file.name.split('_lr_model_info.json')[0]
## Obtain the result labels
result_labels = [get_label(f) for f in result_files if ('json' in f.name)]

## Specify output table name and check if it exists already
output_table_name = 'translator_lr_model_results_{}'.format(version_number)
output_table_exists = output_table_name in get_permanent_tables_list()

## Get output labels already saved
existing_outcome_labels = []
if (output_table_exists):
  existing_results_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(output_table_name))
  outcomes_df = existing_results_df.select('outcome').dropDuplicates()
  existing_outcome_labels = [row.outcome for row in outcomes_df.rdd.collect()]

## If table does not exist, load model results from files and write to sandbox table
for label in result_labels:
  ## Continue if label already exists in output table
  if (label in existing_outcome_labels):
    print("Results for '{}' already saved to sandbox table. Skipping...".format(label))
    continue

  ## Load results for current label
  results_df = load_lr_model_and_info(
    outcome_label=label,
    folder_path=lr_models_save_location)

  ## Write/insert results to sandbox table
  if (output_table_name in get_permanent_tables_list()):
    print("Inserting results for '{0}' into table '{1}'...".format(label, output_table_name))
    insert_data_frame_into_delta_table(results_df, output_table_name)
  else:
    print("Writing results for '{0}' to table '{1}'...".format(label, output_table_name))
    write_data_frame_to_sandbox_delta_table(results_df, output_table_name, replace=False)
  
  ## Add outcome label to list of existing outcome labels
  existing_outcome_labels = existing_outcome_labels + [label]

## Otherwise, load from existing table
all_results_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(output_table_name))

# COMMAND ----------

# DBTITLE 1,Generate final knowledge graph
## Load raw results from sandbox table
table_name = 'translator_lr_model_results_{}'.format(version_number)
raw_results_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(table_name))
all_col_names = raw_results_df.columns

## Load medication feature definitions
# root_folder = "abfss://redap-isb-all@stgredapuserrw.dfs.core.windows.net/rdp_phi_sandbox/hadlock_common/"
# med_cc_dictionary_file = root_folder + "clinical_concepts/translator_medication_feature_definitions_{}.json".format(version_number)
# translator_medication_feature_definitions = spark.read.json(med_cc_dictionary_file)
# translator_medication_feature_definitions = translator_medication_feature_definitions.rdd.collect()[0].asDict()

## Get all feature definitions
all_feature_definitions = translator_condition_feature_definitions
all_feature_definitions.update(translator_medication_feature_definitions)
all_feature_definitions.update(translator_lab_feature_definitions)

## Define convenience functions
def get_concept_property(label, prop):
  feature_info = all_feature_definitions.get(label, None)
  if not(feature_info is None):
    feature_info = feature_info.asDict() if not(isinstance(feature_info, dict)) else feature_info
  feature_prop = feature_info[prop] if not(feature_info is None) else None
  return feature_prop
get_concept_property_udf = F.udf(get_concept_property, StringType())

###################################################################
## Modified by Qi WEI
## Define columns for final table
#####################################
## Feature name related columns
subject_id_col = get_concept_property_udf('feature', F.lit('id')).alias('subject')
subject_name_col = get_concept_property_udf('feature', F.lit('name')).alias('subject_name')
subject_category_col = get_concept_property_udf('feature', F.lit('category')).alias('subject_category')
## Log odds ratio related columns
predicate_col = F.when(F.col('model_coefficient') >= 0, F.lit('biolink:associated_with_increased_likelihood_of')).otherwise(F.lit('biolink:associated_with_decreased_likelihood_of')).alias('predicate')
## Outcome related columns
object_id_col = get_concept_property_udf('outcome', F.lit('id')).alias('object')
object_name_col = get_concept_property_udf('outcome', F.lit('name')).alias('object_name')
object_category_col = get_concept_property_udf('outcome', F.lit('category')).alias('object_category')
## provider related columns
relation_col = F.lit('RO:0003308').alias('relation')
provided_by_col = F.lit('EHR Risk Provider (Multiomics)').alias('provided_by')
provided_date_col = F.lit(version_number).alias('provided_date') ## It shouldn't be hard-codes......fixed by using the version number I defined, in the future could use the latest date
category_col = F.lit('biolink:Association').alias('category')
classifier_col = F.lit('Logistic Regression').alias('classifier')
## Following are results columns
auc_roc_col = F.col('auc_roc')
feature_importance_col = F.lit('NA').alias('feature_importance')
feature_coefficient_col = F.col('model_coefficient').alias('feature_coefficient')
lower_CI_col = F.col('lower_CI').alias('lower_confidence_bound')
upper_CI_col = F.col('upper_CI').alias('upper_confidence_bound')
p_value_col = F.col('p_value').alias('p_value')

## Commented out in Ryan's orginal code
## Following codes are used to obtain the original magnitude of the positive and negative patients count
## Used pyspark.sql.functions.round function
## Round the given value to scale decimal places using HALF_UP rounding mode if scale >= 0 or at integral part when scale < 0.
## e.g. 1536 -> 1500
pos_patient_count_col = (F.round(F.col('positive_patient_count')/10, -1)*10).cast(IntegerType()).alias('positive_patient_count')
neg_patient_count_col = (F.round(F.col('negative_patient_count')/10, -1)*10).cast(IntegerType()).alias('negative_patient_count')
## Following codes are used to obtain the log magnitude of the positive and negative patients count
# pos_patient_count_col = F.floor(F.log10('positive_patient_count')).cast(IntegerType()).alias('log_positive_patient_count')
# neg_patient_count_col = F.floor(F.log10('negative_patient_count')).cast(IntegerType()).alias('log_negative_patient_count')

## Or convert it in the parser?
## Convert it to the actual text instead of just number
## e.g. Sample size with atrial fibrillation, approximately: 500
## Template: Sample size with {}, approximately: x_patient_count

## Generate final kg table
final_columns = [
  subject_id_col, subject_name_col, subject_category_col, predicate_col, object_id_col, object_name_col,
  object_category_col, relation_col, provided_by_col, provided_date_col, category_col, classifier_col,
  auc_roc_col, feature_importance_col, feature_coefficient_col, lower_CI_col, upper_CI_col, p_value_col,
  pos_patient_count_col, neg_patient_count_col]
final_kg_df = raw_results_df.select(final_columns)

## Write results to sandbox table
output_table_name = 'translator_final_kg_{}'.format(version_number)
write_data_frame_to_sandbox_delta_table(final_kg_df, output_table_name, replace=True)

# COMMAND ----------

# DBTITLE 1,Generate final nodes and edges files (csv)
## Load knowledge graph
table_name = 'translator_final_kg_{}'.format(version_number)
kg_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(table_name)) \
  .where(~F.col('subject').isNull())
## Selection columns
select_col_aliases = {'subject': 'id', 'subject_name': 'name', 'subject_category': 'category'}
select_cols = [F.col(k).alias(v) for k, v in select_col_aliases.items()]
select_cols = select_cols + [F.lit('NA').alias('xref'), 'provided_by']

## Generate nodes table
nodes_df = kg_df.select(select_cols).dropDuplicates()
## Generate edges table
## Notice: if need the log magnitude of the patient count then uncomment and change the log part
edges_col_selection = [
  'subject', 'predicate', 'object', 'relation', 'provided_by', 'provided_date', 'category', 'classifier',
  'auc_roc', 'feature_importance', 'feature_coefficient', 'lower_confidence_bound', 'upper_confidence_bound', 'p_value',
  'positive_patient_count', 'negative_patient_count']
edges_df = kg_df.select(edges_col_selection)

## Add duplicate edges for other predicates
predicate_map = {
  'biolink:associated_with_increased_likelihood_of': 'biolink:associated_with_risk_for',
  'biolink:associated_with_decreased_likelihood_of': 'biolink:negatively_associated_with_risk_for'}

## The predicate function
def get_other_predicate(p): return predicate_map.get(p, 'biolink:associated_with_increased_likelihood_of')
get_other_predicate_udf = F.udf(get_other_predicate, StringType())

other_predicates_col = [get_other_predicate_udf(c).alias(c) if (c == 'predicate') else c for c in edges_col_selection]
edges_duplicate_df = kg_df.select(other_predicates_col)

# Get union of all edges
edges_union_df = edges_df.union(edges_duplicate_df)

# COMMAND ----------

# DBTITLE 1,Write edges and nodes files
# # Write nodes and edges files
# output_folder_name = 'lr_models_2022_06_01'
# lr_models_save_location = "abfss://redap-isb-all@stgredapuserrw.dfs.core.windows.net/FileStore/shared_uploads/hadlock_common/translator/{}/".format(output_folder_name)
# nodes_df.orderBy('name').write.option("header", "true").option("delimiter", "\t").csv(lr_models_save_location + 'ehr_risk_nodes_data_2022_06_01.tsv')
# edges_df.orderBy('object', F.col('feature_coefficient').desc(), 'predicate') \
#   .write.option("header", "true").option("delimiter", "\t").csv(lr_models_save_location + 'ehr_risk_edges_data_2022_06_01.tsv')

## Write results to sandbox table
output_table_name = 'ehr_risk_nodes_data_{}'.format(version_number)
write_data_frame_to_sandbox_delta_table(nodes_df, output_table_name, replace=True)

output_table_name = 'ehr_risk_edges_data_{}'.format(version_number)
write_data_frame_to_sandbox_delta_table(edges_union_df, output_table_name, replace=True)

# COMMAND ----------

print("Please check the output size:")
print(nodes_df.count())

display(nodes_df)

# COMMAND ----------

print("Please check the output size:")
print(edges_union_df.count())

display(edges_union_df)
## Check the data type
#edges_union_df.dtypes

# COMMAND ----------

# DBTITLE 1,Get edges dataframe with human readable colnames
from pyspark.sql.functions import concat, col, lit, when

# Load knowledge graph
table_name = 'translator_final_kg_{}'.format(version_number)
kg_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(table_name)) \
  .where(~F.col('subject').isNull())
# Selection columns
select_col_aliases = {'subject': 'id', 'subject_name': 'name', 'subject_category': 'category'}
select_cols = [F.col(k).alias(v) for k, v in select_col_aliases.items()]
select_cols = select_cols + [F.lit('NA').alias('xref'), 'provided_by']

## Convert it to the actual text instead of just number
## e.g. Sample size with atrial fibrillation, approximately: 500
## Template: Sample size with {}, approximately: x_patient_count
kg_df2 = kg_df.withColumns({'positive_patient_text': concat(lit("Sample size with "), col("subject_name"), lit(", approximately: "), col("positive_patient_count")),\
   'negative_patient_text': concat(lit("Sample size with "), col("subject_name"), lit(", approximately: "), col("negative_patient_count"))})

## Fix the p value == 0 when really small issue
kg_df3 = kg_df2.withColumn("p_value_readable", \
              when(kg_df2["p_value"] == 0, lit(0.0001)).otherwise(kg_df2["p_value"]))

# Generate edges table
## Notice: if need the log magnitude of the patient count then uncomment and change the log part
edges_col_selection = [
  'subject', 'subject_name', 'predicate', 'object', 'object_name', 'relation', 'provided_by', 'provided_date', 'category', 'classifier',
  'auc_roc', 'feature_importance', 'feature_coefficient', 'lower_confidence_bound', 'upper_confidence_bound', 'p_value', 'p_value_readable',
  'positive_patient_count', 'negative_patient_count', 'positive_patient_text', 'negative_patient_text']
edges_df = kg_df3.select(edges_col_selection)

# # Add duplicate edges for other predicates
predicate_map = {
  'biolink:associated_with_increased_likelihood_of': 'biolink:associated_with_risk_for',
  'biolink:associated_with_decreased_likelihood_of': 'biolink:negatively_associated_with_risk_for'}

# ## The predicate function
def get_other_predicate(p): return predicate_map.get(p, 'biolink:associated_with_increased_likelihood_of')
get_other_predicate_udf = F.udf(get_other_predicate, StringType())

other_predicates_col = [get_other_predicate_udf(c).alias(c) if (c == 'predicate') else c for c in edges_col_selection]
edges_duplicate_df = kg_df3.select(other_predicates_col)

# Get union of all edges
edges_union_readable_df = edges_df.union(edges_duplicate_df)

print("Please check the output size:")
print(edges_union_readable_df.count())

display(edges_union_readable_df)