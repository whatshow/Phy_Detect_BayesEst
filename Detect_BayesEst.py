import torch
import numpy as np

# warning messages
MSG_INIT_CONSTELLATION_DIM_HIGH = "[constellation] cannot have over 1 dimension.";
MSG_INIT_CONSTELLATION_NOTSUPPORTTED = "[constellation] is not supported.";
MSG_INIT_MINVAR_TYPE_WRONG = "[min_var] can only be an integer or a float.";

class Detect_BayesEst:
    min_var = np.finfo(np.float_).eps;          # the default minimal variance is 2.22e-16
    
    '''
    init
    @constellation: the constellation points value, it should be 1 dimentional
    @min_var:       scalar, the minimal variance in the output
    '''
    def __init__(self, constellation, *, min_var = None):
        # constellation
        # tensor
        if isinstance(constellation, torch.Tensor):
            if constellation.dim() > 1:
                raise Exception(MSG_INIT_CONSTELLATION_DIM_HIGH);
        # numpy & python types
        else:
            try:
                self.constellation = np.asarray(constellation);
                if self.constellation.ndim > 1:
                    raise Exception(MSG_INIT_CONSTELLATION_DIM_HIGH);
            except:
                raise Exception(MSG_INIT_CONSTELLATION_NOTSUPPORTTED);
        
        # min_var
        if min_var is not None:
            if not isinstance(min_var, int) and not isinstance(min_var, float):
                raise Exception(MSG_INIT_MINVAR_TYPE_WRONG);
            else:
                self.min_var = min_var;
        
    '''
    Baesian estimate new mean & variance
    @mean:      vector, the observation of received signals
    @var:       scalar or vector of the variance of y
    @decoding:  scalar, if it is decoding, we just output the most possible x
    '''
    def detect(self, mean, var, *, decoding = False):
        # input type check
        try:
            mean = np.asarray(mean);
        except:
            raise Exception("[mean] can't be converted into a numpy array.");
        try:
            var = np.asarray(var);
        except:
            raise Exception("[var] can't be converted into a numpy array.");
        # make sure inputs have the same dimension
        if var.ndim != 0 and var.shape != mean.shape:
            raise Exception("[var] should be a scalar or with the same shape with the mean.");
        
        # adjust the shape of mean, var, constellation
        var = np.expand_dims(var, -1);
        mean = np.expand_dims(mean, -1);
        mean_shape = list(mean.shape);
        constellation_extended_shape = mean_shape;
        constellation_extended_shape[-1] = 1;
        constellation_extended = np.tile(self.constellation, constellation_extended_shape);
        
        # Estimate P(x|y) using Gaussian distribution
        pxyPdfExpPower = -1/(2*var)*np.square(np.abs(mean - constellation_extended)); # Here, we use numpy array broadcasting (the last dimension of mean is 1)
        pxypdfExpNormPower = pxyPdfExpPower - np.expand_dims(np.max(pxyPdfExpPower, -1), -1);
        pxyPdf = np.exp(pxypdfExpNormPower);
        # Calculate the coefficient of every possible x to make the sum of all possbilities is 1
        pxyPdfNorm = pxyPdf/np.expand_dims(np.sum(pxyPdf, -1), -1);
        
        # calculate the mean & variance
        pxyMean = None;
        pxyVar = None;
        if decoding is True:
            pxyMean = self.constellation[np.argmax(pxyPdfNorm,-1)];
        else:
            pxyMean = np.sum(pxyPdfNorm*constellation_extended, -1);
            pxyVar = np.sum(np.square(np.abs(np.expand_dims(pxyMean, -1) - constellation_extended)) * pxyPdfNorm, -1);
            # limit the minimal variance
            pxyVar = np.clip(pxyVar, self.min_var, None);
        
        # return
        if decoding is True:
            return pxyMean;
        else:
            return pxyMean, pxyVar;