import numpy as np
import scipy.io as sci
import sys 
sys.path.append("../..") 
import Detect_BayesEst;

###############################################################################
# Bayesian Estimation
# @author: Alva Kosasih
###############################################################################
'''
calculate the probability based on the Gaussian distribution
'''
def calculate_pyx(mean, var, constellation_expanded):
    user_num = mean.shape[-1];
    
    constellation_expanded_transpose = np.repeat(constellation_expanded.transpose(0,2,1), user_num, axis=1)
    arg_1 = np.square(np.abs(constellation_expanded_transpose - np.expand_dims(mean,2)))
    log_pyx = (-1 * arg_1)/(2*np.expand_dims(var,2))
    log_pyx = log_pyx - np.expand_dims(np.max(log_pyx,2),2)
    p_y_x = np.exp(log_pyx)
    p_y_x = p_y_x/(np.expand_dims(np.sum(p_y_x, axis=2),2))
    
    return p_y_x

'''
calculate the mean & variance based on a given probability
'''
def calculate_mean_var(pyx, constellation_expanded):
    user_num = pyx.shape[-2];
    
    constellation_expanded_transpose = np.repeat(constellation_expanded.transpose(0,2,1), user_num, axis=1)
    mean = np.matmul(pyx, constellation_expanded)
    var = np.square(np.abs(constellation_expanded_transpose - mean))
    var = np.multiply(pyx, var) 
    var = np.sum(var, axis=2)
    
    return np.squeeze(mean), var
###############################################################################


###############################################################################
# main
###############################################################################
print("Testing starts:");
# load matlab data (1st test data)
matlab_data                 = sci.loadmat("Detect_BayesEst_InOut.mat");
mat_constellation           = np.squeeze(matlab_data['xpool']);
mat_mu_A_B                  = np.squeeze(matlab_data['mu_A_B']);
mat_zigma_A_B               = np.squeeze(matlab_data['zigma_A_B']);
mat_mu_B                    = np.squeeze(matlab_data['mu_B']);
mat_var_B                   = np.squeeze(matlab_data['var_B']);
# load python data (2nd test data)
py_constellation            = np.load("Detect_BayesEst_InOut_constellation.npy");
py_constellation_expanded   = np.load("Detect_BayesEst_InOut_constellation_expanded.npy");
py_mu_A_B                   = np.load("Detect_BayesEst_InOut_mean.npy");
py_zigma_A_B                = np.load("Detect_BayesEst_InOut_var.npy");
py_pyx                      = np.load("Detect_BayesEst_InOut_pyx.npy");
py_mu_B                     = np.load("Detect_BayesEst_InOut_mean_est.npy");
py_var_B                    = np.load("Detect_BayesEst_InOut_var_est.npy");

# test whether Alva's old code is correct
print("- test whether Alva's old code is correct or not");
pyx = calculate_pyx(py_mu_A_B, py_zigma_A_B, py_constellation_expanded);
print("  [Result] the difference of pyx is %.16f"%sum(sum(sum(pyx - py_pyx))));
mu_B, var_B = calculate_mean_var(pyx, py_constellation_expanded);
print("  [Result] the difference of estimated mean is %.16f"%sum(sum(mu_B - py_mu_B)));
print("  [Result] the difference of estimated variance is %.16f"%sum(sum(var_B - py_var_B)));

# test whether Detect_BayesEst.py is correct or not
print("- test whether Detect_BayesEst.py is correct or not");
bayes_est_detector = Detect_BayesEst(py_constellation);
pxyMean, pxyVar = bayes_est_detector.detect(py_mu_A_B, py_zigma_A_B);
print("  [Result] the difference of estimated mean is %.16f"%sum(sum(pxyMean - py_mu_B)));
print("  [Result] the difference of estimated variance is %.16f"%sum(sum(pxyVar - py_var_B)));

# test whether Detect_BayesEst.py suits 1D data
print("- test whether Detect_BayesEst.py suits 1D data");
bayes_est_detector = Detect_BayesEst(mat_constellation);
pxyMean, pxyVar = bayes_est_detector.detect(mat_mu_A_B, mat_zigma_A_B, decoding=False);
print("  [Result] the difference of estimated mean is %.16f"%sum(abs(pxyMean - mat_mu_B)));
print("  [Result] the difference of estimated variance is %.16f"%sum(abs(pxyVar - mat_var_B)));
pxyMean = bayes_est_detector.detect(mat_mu_A_B, mat_zigma_A_B, decoding=True);
print("  [Result] the difference of estimated mean(decoding) is %.16f"%(sum(abs(pxyMean - mat_mu_B)) - 12.825280));
