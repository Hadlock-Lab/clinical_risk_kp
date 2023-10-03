# Databricks notebook source
# MAGIC %md
# MAGIC ## Notes

# COMMAND ----------

# DBTITLE 1,Load CEDA Tools
# MAGIC %run
# MAGIC "/Users/qi.wei1@providence.org/CEDA-tool-test/CEDA Tools v1.1 test/load_ceda_api"

# COMMAND ----------

# DBTITLE 1,Sanity check the file path
# display(dbutils.fs.ls("abfss://redap-isb-all@stgredapuserrw.dfs.core.windows.net/rdp_phi_sandbox/hadlock_common/translator/"))

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

# DBTITLE 1,Create location to save logistic regression models
# Specify folder location to save logistic regresssion models
output_folder_name = 'all_cond_lr_models_2023_08_10_CI_pValue/'
lr_models_save_location = "abfss://redap-isb-all@stgredapuserrw.dfs.core.windows.net/rdp_phi_sandbox/hadlock_common/translator/{}".format(output_folder_name)

# One-time creation of output folder for logistic regression models
try:
  dbutils.fs.ls(lr_models_save_location)
  print("Output folder already exists:\n" + lr_models_save_location)
except:
  dbutils.fs.mkdirs(lr_models_save_location)
  print("New output folder created:\n" + lr_models_save_location)

# Show labels of saved logistic regression models
print("\nDiseases for which results exist:")
result_files = dbutils.fs.ls(lr_models_save_location)
file_count = 0
for file in result_files:
  print(file)
  if ('json' in file.name):
    file_count += 1
    print(f"{file_count}.\t{file.name.split('_lr_model_info.json')[0]}")

# COMMAND ----------

# DBTITLE 1,Conditions/Phenotypes for which to run models
##
list_not_working = ['alstrom_syndrome', 'white_matter_disorder_with_cadasil', 'ullrich_congenital_muscular_dystrophy', 'bethlem_myopathy', 'cockayne_syndrome', 'congenital_erythropoietic_porphyria', 'parkes_weber_syndrome', 'kimura_disease', 'ehlers_danlos_syndrome_kyphoscoliotic_type', 'ehlers_danlos_syndrome_procollagen_proteinase_deficient', 'hyperleucinemia', 'hereditary_insensitivity_to_pain_with_anhidrosis', 'familial_malignant_melanoma_of_skin', 'fatal_familial_insomnia',''] 

outcome_conditions = ['cystic_fibrosis', 'ehlers_danlos_syndrome', 'diabetes_mellitus_type_2', 'chronic_heart_failure', 'chronic_hypertension', 'osteoarthritis', 'chronic_obstructive_pulmonary_disease', 'chronic_ischemic_heart_disease', 'chronic_kidney_disease', 'osteoporosis', 'hyperlipidemia', 'atrial_fibrillation', 'chronic_hepatitis_b_c', 'obesity_disorder', 'hiv', 'psoriatic_arthritis', 'psoriasis', 'rheumatoid_arthritis', 'narcolepsy', 'fabry_disease', 'primary_biliary_cholangitis', 'cerebral_autosomal_dominant_arteriopathy_with_subcortical_infarcts_and_leukoencephalopathy', 'hypermobile_ehlers_danlos_syndrome', 'vascular_ehlers_danlos_syndrome', 'classical_ehlers_danlos_syndrome', 'inclusion_body_myositis', 'castleman_disease', 'hemophilia', 'spinal_muscular_atrophy', 'retinal_dystrophy', 'familial_x_linked_hypophosphatemic_vitamin_d_refractory_rickets', 'adrenal_cushing_syndrome', 'pulmonary_hypertensive_arterial_disease', 'brugada_syndrome', 'erythropoietic_protoporphyria', 'guillain_barre_syndrome', 'tetralogy_of_fallot', 'discordant_ventriculoarterial_connection', 'focal_dystonia', 'marfan_syndrome', 'non_hodgkin_lymphoma', 'retinitis_pigmentosa', 'gelineau_syndrome', 'multiple_myeloma', 'alpha_1_antitrypsin_deficiency', 'congenital_diaphragmatic_hernia', 'juvenile_idiopathic_arthritis', 'neurofibromatosis_type_1', 'congenital_atresia_of_esophagus', 'polycythemia_vera', 'hereditary_motor_and_sensory_neuropathy', 'polycystic_kidney_disease_infantile_type', 'vater_association', 'coffin_lowry_syndrome', 'osler_hemorrhagic_telangiectasia_syndrome', 'dermatitis_herpetiformis', 'congenital_atresia_of_small_intestine', 'congenital_atresia_of_duodenum', 'congenital_aganglionic_megacolon', '22q11_2_deletion_syndrome', 'hereditary_spherocytosis', 'turner_syndrome', 'melas', 'medium_chain_acyl_coenzyme_a_dehydrogenase_deficiency', 'lennox_gastaut_syndrome', 'fragile_x_syndrome', 'stickler_syndrome', 'williams_syndrome', 'von_willebrand_disorder', 'gastroschisis', 'microphthalmos', 'congenital_omphalocele', 'sarcoidosis', 'stargardt_disease', 'glioblastoma_multiforme', 'multiple_endocrine_neoplasia_type_1', 'prader_willi_syndrome', 'alopecia_totalis', 'nephroblastoma', 'duane_syndrome', 'neuroblastoma', 'hodgkin_disease', 'klippel_trenaunay_syndrome', 'whipple_disease', 'incontinentia_pigmenti_syndrome', 'aicardi_syndrome', 'li_fraumeni_syndrome', 'russell_silver_syndrome', 'congenital_livedo_reticularis', 'moebius_syndrome',  'kabuki_make_up_syndrome', 'ondine_curse', 'job_syndrome', 'kearns_sayre_syndrome', 'cholestanol_storage_disease', 'cogan_syndrome', 'alcohol_abuse', 'alcoholic_liver_disease', 'allergic_rhinitis', 'alzheimers_disease', 'ankylosing_spondyloarthritis', 'arteriosclerotic_of_artery_other', 'asthma', 'attention_deficit_hyperactivity_disorder', 'autoimmune_liver_disease', 'carotid_atherosclerosis', 'celiac_disease', 'cerebral_atherosclerosis', 'chronic_depression', 'chronic_fatigue_syndrome', 'chronic_hepatitis_b', 'chronic_hepatitis_c', 'chronic_kidney_disease_i_to_iv', 'chronic_kidney_disease_v', 'complex_regional_pain_syndrome_', 'copd_chronic_obtructive_lung_disease', 'crohns_disease_', 'diabetes_type_1_', 'diabetes_type_2', 'endometriosis_', 'excema', 'fibromyalgia_', 'gerd', 'gout', 'graves_disease', 'heart_failure', 'heart_valve_disease', 'herpes_simplex', 'huntingtons_disease_', 'hyperglycemia_', 'hyperthyroidism', 'menieres_disease_', 'multiple_sclerosis_', 'narcolepsy_', 'nonalcoholic_steatohepatitis', 'obesity', 'obsessive_compulsive_disorder', 'parkinsons_disease_', 'peripheral_arterial_disease', 'peripheral_neuropathy_', 'polycystic_ovarian_disease_', 'posttraumatic_stress_disorder', 'psoriasis_without_psoriatic_arthritis', 'pulmonary_hypertension', 'rosacea', 'scoliosis', 'systemic_lupus_erethematosus', 'trigeminal_neuralgia_', 'ulcerative_colitis', 'vascular_dementia', 'vasculitis', 'vitamain_b12_defiency']

# COMMAND ----------

# DBTITLE 1,Remove duplicates in entries and sort them
outcome_conditions = list(set(outcome_conditions))
# print("Please Use following numbers to guide on how many threads needed")
print(len(outcome_conditions))
outcome_conditions.sort()

# COMMAND ----------

# DBTITLE 1,Print out the colnames of the training dataset
train_df = spark.sql("""select * from rdp_phi_sandbox.hadlock_ml_features_encounter_{}""".format(version_number))
train_df.columns

# COMMAND ----------

# DBTITLE 1,Use the multi_threading pool to run those LR models in parallel, current set to be 240 threads in parallel
from multiprocessing.pool import ThreadPool
pool = ThreadPool(1)

def multi_threading_LR(x):
  # Specify outcome label and source data table
  outcome_label = x
  ml_enc_table_name = 'hadlock_ml_features_encounter_{}'.format(version_number)
  # Run if model results do not already exist
  try:
    # Check if model results already exist
    info_file_name = '{}_lr_model_info.json'.format(outcome_label)
    dbutils.fs.ls(lr_models_save_location + info_file_name)
    print("Results exist for '{}'. Skipping...".format(outcome_label))  
  except:
    # Run logistic regression model
    print("Running model for '{}'...".format(outcome_label))
    lr, info, summary = run_logistic_regression(
      ml_encounter_table = ml_enc_table_name, outcome_column=outcome_label)
    
    print("Finished model for '{}', now move to save the output".format(outcome_label))
    ## Save model results and information
    save_lr_model_and_info(
      lr_model= lr,
      model_info= info,
      outcome_label= outcome_label,
      folder_path= lr_models_save_location)
    print("Finished saving result for model for '{}'...".format(outcome_label))

## Run the multi_threading functions in 30 times faster
pool.map(multi_threading_LR, outcome_conditions)
