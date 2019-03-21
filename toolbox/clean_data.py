
import numpy as np
from toolbox.implementations import build_poly, ridge_regression
from toolbox.cross_validation import standardize

ind_DER_mass_MMC = 0
ind_DER_mass_transverse_met_lep = 1
ind_DER_mass_vis = 2
ind_DER_pt_h   = 3
ind_DER_deltaeta_jet_jet = 4
ind_DER_mass_jet_jet = 5
ind_DER_prodeta_jet_jet = 6
ind_DER_deltar_tau_lep = 7
ind_DER_pt_tot   = 8
ind_DER_sum_pt   = 9
ind_DER_pt_ratio_lep_tau   = 10
ind_DER_met_phi_centrality   = 11
ind_DER_lep_eta_centrality   = 12
ind_PRI_tau_pt   = 13
ind_PRI_tau_eta   = 14
ind_PRI_tau_phi   = 15
ind_PRI_lep_pt   = 16
ind_PRI_lep_eta = 17
ind_PRI_lep_phi = 18
ind_PRI_met   = 19
ind_PRI_met_phi   = 20
ind_PRI_met_sumet   = 21
ind_PRI_jet_num   = 22
ind_PRI_jet_leading_pt   = 23
ind_PRI_jet_leading_eta   = 24
ind_PRI_jet_leading_phi   = 25
ind_PRI_jet_subleading_pt   = 26
ind_PRI_jet_subleading_eta   = 27
ind_PRI_jet_subleading_phi   = 28
ind_PRI_jet_all_pt   = 29


def split_data_jetnum(tX):
    # Extract indices from dataset only jet_num = 0 // ravel() is used to transform into a single vector the indices
    ind_jet0 = np.argwhere((tX[:,ind_PRI_jet_num] == 0)).ravel()
    # Extract indices from dataset only jet_num = 3 // ravel() is used to transform into a single vector the indices
    ind_jet1 = np.argwhere((tX[:,ind_PRI_jet_num] == 1)).ravel()
    # Extract indices from dataset only jet_num = 3 or 2// ravel() is used to transform into a single vector the indices
    ind_jet2 = np.argwhere((tX[:,ind_PRI_jet_num] == 2)).ravel()

    ind_jet3 = np.argwhere((tX[:,ind_PRI_jet_num] == 3)).ravel()


    # Create new dataset extracting only jet_num = 0
    datajet0 = tX[ind_jet0]
    colToDelete_jet0 = [ind_PRI_jet_num,ind_DER_deltaeta_jet_jet, ind_DER_mass_jet_jet, ind_DER_prodeta_jet_jet, ind_DER_lep_eta_centrality, ind_PRI_jet_subleading_pt, ind_PRI_jet_subleading_eta, ind_PRI_jet_subleading_phi, ind_PRI_jet_leading_pt, ind_PRI_jet_leading_eta, ind_PRI_jet_leading_phi, ind_PRI_jet_all_pt]
    datajet0 = np.delete(datajet0, colToDelete_jet0 , axis=1)
    # Create new dataset extracting only jet_num = 1
    datajet1 = tX[ind_jet1]
    colToDelete_jet1 = [ind_PRI_jet_num,ind_DER_deltaeta_jet_jet, ind_DER_mass_jet_jet, ind_DER_prodeta_jet_jet, ind_DER_lep_eta_centrality, ind_PRI_jet_subleading_pt, ind_PRI_jet_subleading_eta, ind_PRI_jet_subleading_phi, ind_PRI_jet_all_pt ]
    datajet1 = np.delete(datajet1, colToDelete_jet1 , axis=1)
    # Create new dataset extracting only jet_num = 2
    datajet2 = tX[ind_jet2]
    colToDelete_jet2 = [ind_PRI_jet_num]
    datajet2 = np.delete(datajet2, colToDelete_jet2 , axis=1)

    # Create new dataset extracting only jet_num = 3
    datajet3 = tX[ind_jet3]
    colToDelete_jet3 = [ind_PRI_jet_num]
    datajet3 = np.delete(datajet3, colToDelete_jet3 , axis=1)

    return datajet0, datajet1, datajet2, datajet3, ind_jet0, ind_jet1, ind_jet2, ind_jet3

def extract_wrong_values(datajet):
    ind_wrongjet = np.argwhere(datajet[:,ind_DER_mass_MMC] == -999.000).ravel()
    ind_rightjet = np.argwhere(datajet[:,ind_DER_mass_MMC] != -999.000).ravel()

    wrongjet = datajet[ind_wrongjet]
    rightjet = datajet[ind_rightjet]

    return rightjet, wrongjet, ind_rightjet, ind_wrongjet

def create_training_DERmass(rightjet):
    y_rightjet = rightjet[:, ind_DER_mass_MMC]
    tx_rightjet = np.delete(rightjet, ind_DER_mass_MMC, axis=1)

    return tx_rightjet, y_rightjet

def create_correction_model(tx_rightjet, y_rightjet, degree = 2, crossterm=False, lambda_ = 0.00001):
    tx_rightjet_poly = build_poly(tx_rightjet, degree, crossterm)
    tx_rightjet_poly,mean_txjet,std_txjet = standardize(tx_rightjet_poly)
    tx_rightjet_poly[:,0] = np.ones(len(tx_rightjet_poly))

    # Process Y of jet0 (DET_mass_MMC)
    stand_y_rightjet,mean_yjet,std_yjet = standardize(y_rightjet)

    # Train/Compute the Model for Jet = 0
    loss_jet, w_jet = ridge_regression(stand_y_rightjet, tx_rightjet_poly, lambda_)

    return w_jet, mean_txjet, std_txjet, mean_yjet, std_yjet

def prepare_correct_values(wrongjet, mean_txjet, std_txjet, degree = 2, crossterm=False, lambda_ = 0.00001):
    tx_wrongjet = np.delete(wrongjet, ind_DER_mass_MMC, axis=1)
    tx_wrongjet_poly = build_poly(tx_wrongjet, degree, crossterm)
    tx_wrongjet_poly,_,_ = standardize(tx_wrongjet_poly,mean_txjet,std_txjet)
    tx_wrongjet_poly[:,0] = np.ones(len(tx_wrongjet_poly)) ####### ADD to FUNCTION

    return tx_wrongjet_poly

def compute_correct_values(tx_wrongjet_poly, w_jet, mean_yjet, std_yjet):
    y_predMASS_jet = tx_wrongjet_poly.dot(w_jet)
    y_predMASS_jet = (y_predMASS_jet*std_yjet) + mean_yjet

    return y_predMASS_jet

def replace_correct_values(y_predMASS_jet, wrongjet, datajet, ind_wrongjet):
    wrongjet[:, ind_DER_mass_MMC] = y_predMASS_jet
    datajet[ind_wrongjet] = wrongjet

    return datajet
