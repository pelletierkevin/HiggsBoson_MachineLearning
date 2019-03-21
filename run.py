import matplotlib.pyplot as plt
import numpy as np
from toolbox.proj1_helpers import  load_csv_data, predict_labels, create_csv_submission
from toolbox.implementations import ridge_regression, build_poly
from toolbox.cross_validation import standardize
from toolbox.clean_data import split_data_jetnum, extract_wrong_values, create_training_DERmass, create_correction_model, prepare_correct_values, compute_correct_values, replace_correct_values

DATA_FOLDER = 'data'
DATA_TRAIN_PATH = 'data/train.csv'
DATA_TEST_PATH = 'data/test.csv'

y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

############ DATA CLEANING #############

####### ON TRAINING SET

# Split the training data into 4 parts (depending on Jetnum value) and the associated indices
datajet0, datajet1, datajet2,datajet3, ind_jet0, ind_jet1, ind_jet2, ind_jet3 = split_data_jetnum(tX)

#### ---- DER_mass_MMC wrong values : Training a model to predict correct values of ind_DER_mass_MMC

# Extract/Split values of each dataset, to get Input data and Output data (DEr_mass_MMC)
rightjet0, wrongjet0, ind_rightjet0, ind_wrongjet0 = extract_wrong_values(datajet0)
rightjet1, wrongjet1, ind_rightjet1, ind_wrongjet1 = extract_wrong_values(datajet1)
rightjet2, wrongjet2, ind_rightjet2, ind_wrongjet2 = extract_wrong_values(datajet2)
rightjet3, wrongjet3, ind_rightjet3, ind_wrongjet3 = extract_wrong_values(datajet3)

tx_rightjet0, y_rightjet0 = create_training_DERmass(rightjet0)
tx_rightjet1, y_rightjet1 = create_training_DERmass(rightjet1)
tx_rightjet2, y_rightjet2 = create_training_DERmass(rightjet2)
tx_rightjet3, y_rightjet3 = create_training_DERmass(rightjet3)

# Train the models for correct values of DER_mass_MMC
wMASS_jet0, mean_txMASSjet0, std_txMASSjet0, mean_yMASSjet0, std_yMASSjet0 = create_correction_model(tx_rightjet0, y_rightjet0, degree = 2, crossterm=False, lambda_ = 0.00001)
wMASS_jet1, mean_txMASSjet1, std_txMASSjet1, mean_yMASSjet1, std_yMASSjet1 = create_correction_model(tx_rightjet1, y_rightjet1, degree = 1, crossterm=False, lambda_ = 0.00001)
wMASS_jet2, mean_txMASSjet2, std_txMASSjet2, mean_yMASSjet2, std_yMASSjet2 = create_correction_model(tx_rightjet2, y_rightjet2, degree = 1, crossterm=False, lambda_ = 0.00001)
wMASS_jet3, mean_txMASSjet3, std_txMASSjet3, mean_yMASSjet3, std_yMASSjet3 = create_correction_model(tx_rightjet3, y_rightjet3, degree = 1, crossterm=False, lambda_ = 0.00001)

# Prepare the wrong values to correct them (Polynomial, Bias, Standardization)
tx_wrongjet0_poly = prepare_correct_values(wrongjet0, mean_txMASSjet0, std_txMASSjet0, degree = 2, crossterm=False, lambda_ = 0.00001)
tx_wrongjet1_poly = prepare_correct_values(wrongjet1, mean_txMASSjet1, std_txMASSjet1, degree = 1, crossterm=False, lambda_ = 0.00001)
tx_wrongjet2_poly = prepare_correct_values(wrongjet2, mean_txMASSjet2, std_txMASSjet2, degree = 1, crossterm=False, lambda_ = 0.00001)
tx_wrongjet3_poly = prepare_correct_values(wrongjet3, mean_txMASSjet3, std_txMASSjet3, degree = 1, crossterm=False, lambda_ = 0.00001)

# Apply the ML prediction to get the "correct" values for each row
y_predMASS_jet0 = compute_correct_values(tx_wrongjet0_poly, wMASS_jet0, mean_yMASSjet0, std_yMASSjet0)
y_predMASS_jet1 = compute_correct_values(tx_wrongjet1_poly, wMASS_jet1, mean_yMASSjet1, std_yMASSjet1)
y_predMASS_jet2 = compute_correct_values(tx_wrongjet2_poly, wMASS_jet2, mean_yMASSjet2, std_yMASSjet2)
y_predMASS_jet3 = compute_correct_values(tx_wrongjet3_poly, wMASS_jet3, mean_yMASSjet3, std_yMASSjet3)

# Replace wrong values of DER_mass_MMC with newly computed correct values
datajet0 = replace_correct_values(y_predMASS_jet0, wrongjet0, datajet0, ind_wrongjet0)
datajet1 = replace_correct_values(y_predMASS_jet1, wrongjet1, datajet1, ind_wrongjet1)
datajet2 = replace_correct_values(y_predMASS_jet2, wrongjet2, datajet2, ind_wrongjet2)
datajet3 = replace_correct_values(y_predMASS_jet3, wrongjet3, datajet3, ind_wrongjet3)


####### ON TEST SET

# Split the test data into 4 parts (depending on Jetnum value) and the associated indices
datatest_jet0, datatest_jet1, datatest_jet2,datatest_jet3, indtest_jet0, indtest_jet1, indtest_jet2, indtest_jet3 = split_data_jetnum(tX_test)

# Extract/Split values of each dataset, to get Input data to compute Output data (DEr_mass_MMC)
_, wrongtest_jet0, _, indtest_wrongjet0 = extract_wrong_values(datatest_jet0)
_, wrongtest_jet1, _, indtest_wrongjet1 = extract_wrong_values(datatest_jet1)
_, wrongtest_jet2, _, indtest_wrongjet2 = extract_wrong_values(datatest_jet2)
_, wrongtest_jet3, _, indtest_wrongjet3 = extract_wrong_values(datatest_jet3)

# Prepare the wrong values to correct them (Polynomial, Bias, Standardization)
txtest_wrongjet0_poly = prepare_correct_values(wrongtest_jet0, mean_txMASSjet0, std_txMASSjet0, degree = 2, crossterm=False, lambda_ = 0.00001)
txtest_wrongjet1_poly = prepare_correct_values(wrongtest_jet1, mean_txMASSjet1, std_txMASSjet1, degree = 1, crossterm=False, lambda_ = 0.00001)
txtest_wrongjet2_poly = prepare_correct_values(wrongtest_jet2, mean_txMASSjet2, std_txMASSjet2, degree = 1, crossterm=False, lambda_ = 0.00001)
txtest_wrongjet3_poly = prepare_correct_values(wrongtest_jet3, mean_txMASSjet3, std_txMASSjet3, degree = 1, crossterm=False, lambda_ = 0.00001)

# Apply the ML prediction to get the "correct" values of DER_mass_MMC for each row
MASSpred_test_jet0 = compute_correct_values(txtest_wrongjet0_poly, wMASS_jet0, mean_yMASSjet0, std_yMASSjet0)
MASSpred_test_jet1 = compute_correct_values(txtest_wrongjet1_poly, wMASS_jet1, mean_yMASSjet1, std_yMASSjet1)
MASSpred_test_jet2 = compute_correct_values(txtest_wrongjet2_poly, wMASS_jet2, mean_yMASSjet2, std_yMASSjet2)
MASSpred_test_jet3 = compute_correct_values(txtest_wrongjet3_poly, wMASS_jet3, mean_yMASSjet3, std_yMASSjet3)

# Replace wrong values of DER_mass_MMC with newly computed correct values
datatest_jet0 = replace_correct_values(MASSpred_test_jet0, wrongtest_jet0, datatest_jet0, indtest_wrongjet0)
datatest_jet1 = replace_correct_values(MASSpred_test_jet1, wrongtest_jet1, datatest_jet1, indtest_wrongjet1)
datatest_jet2 = replace_correct_values(MASSpred_test_jet2, wrongtest_jet2, datatest_jet2, indtest_wrongjet2)
datatest_jet3 = replace_correct_values(MASSpred_test_jet3, wrongtest_jet3, datatest_jet3, indtest_wrongjet3)

############ DATA CLEANED #############


############ HIGGS BOSON PREDICTION MODEL #############

####### TRAINING MODEL <

# Associate each y to the corresponding value of jet num
y_jet0 = y[ind_jet0]
y_jet1 = y[ind_jet1]
y_jet2 = y[ind_jet2]
y_jet3 = y[ind_jet3]

# Prepare the datasets (polynomial, crossterm, standardize,bias)
datajet0_poly = build_poly(datajet0, 8, True, True,True)
datajet0_poly,mean_txj0,std_txj0 = standardize(datajet0_poly)
datajet0_poly[:,0] = np.ones(len(datajet0_poly)) ####### ADD to FUNCTION

datajet1_poly = build_poly(datajet1, 9, True, True,True)
datajet1_poly,mean_txj1,std_txj1 = standardize(datajet1_poly)
datajet1_poly[:,0] = np.ones(len(datajet1_poly)) ####### ADD to FUNCTION

datajet2_poly = build_poly(datajet2, 9, True, True,True)
datajet2_poly,mean_txj2,std_txj2 = standardize(datajet2_poly)
datajet2_poly[:,0] = np.ones(len(datajet2_poly)) ####### ADD to FUNCTION

datajet3_poly = build_poly(datajet3, 9, True, True,True)
datajet3_poly,mean_txj3,std_txj3 = standardize(datajet3_poly)
datajet3_poly[:,0] = np.ones(len(datajet3_poly)) ####### ADD to FUNCTION

# Train the models
loss_j0, w_pred_jet0 = ridge_regression(y_jet0, datajet0_poly, 0.0001)

loss_j1, w_pred_jet1 = ridge_regression(y_jet1, datajet1_poly, 0.0001)

loss_j2, w_pred_jet2 = ridge_regression(y_jet2, datajet2_poly, 0.0001)

loss_j3, w_pred_jet3 = ridge_regression(y_jet3, datajet3_poly, 0.0001)

####### MODEL TRAINED >

####### PREDICTION ON TEST SET <

# Prepare the datasets (polynomial, crossterm, standardize,bias)
datatestjet0_poly = build_poly(datatest_jet0, 8, True, True,True)
datatestjet0_poly,_,_ = standardize(datatestjet0_poly,mean_txj0,std_txj0)
datatestjet0_poly[:,0] = np.ones(len(datatestjet0_poly)) ####### ADD to FUNCTION

datatestjet1_poly = build_poly(datatest_jet1, 9, True, True,True)
datatestjet1_poly,_,_ = standardize(datatestjet1_poly,mean_txj1, std_txj1)
datatestjet1_poly[:,0] = np.ones(len(datatestjet1_poly)) ####### ADD to FUNCTION

datatestjet2_poly = build_poly(datatest_jet2, 9, True, True,True)
datatestjet2_poly,_,_ = standardize(datatestjet2_poly, mean_txj2, std_txj2)
datatestjet2_poly[:,0] = np.ones(len(datatestjet2_poly)) ####### ADD to FUNCTION

datatestjet3_poly = build_poly(datatest_jet3, 9, True, True,True)
datatestjet3_poly,_,_ = standardize(datatestjet3_poly, mean_txj3, std_txj3)
datatestjet3_poly[:,0] = np.ones(len(datatestjet3_poly)) ####### ADD to FUNCTION

# Compute prediction
ytest_predj0 = predict_labels(w_pred_jet0, datatestjet0_poly)

ytest_predj1 = predict_labels(w_pred_jet1, datatestjet1_poly)

ytest_predj2 = predict_labels(w_pred_jet2, datatestjet2_poly)

ytest_predj3 = predict_labels(w_pred_jet3, datatestjet3_poly)


### Merge the output from different dataset to initial y
y_result = np.ones(tX_test.shape[0])*2

y_result[indtest_jet0] = ytest_predj0
y_result[indtest_jet1] = ytest_predj1
y_result[indtest_jet2] = ytest_predj2
y_result[indtest_jet3] = ytest_predj3


create_csv_submission(ids_test, y_result, 'submission_y_predict.csv')
