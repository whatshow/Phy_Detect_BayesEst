# Bayesian Estimation Detection
When we have p(y|x)'s mean and p(y|x)'s variance (the noise power), we can assume p(x|y) fits Gaussian distribution of a mean. Then we test that
mean of all possible x values. After that, we normalise all Gaussian distribution PDF of every x value. (The sum of those must be 1). Then we
use those Gaussian distribution on every x to recalculate the mean and the variance of p(x|y).
* **In another local repositiory, add this module**
```sh
git submodule add git@github.com:USYD-SDN-Lab/Detect_BayesEst.git Modules/Detect_BayesEst
```
Now, you can see a folder `Modules` with `Detect_BayesEst` inside
* **import this module**
    * Matlab
    ```matlab
    addpath("Modules/Detect_BayesEst");
    ```
    * Python
    ```python
    if '.' not in __name__ :
 	    from Modules.Detect_BayesEst.Detect_BayesEst import Detect_BayesEst
    else:
 	    from .Modules.Detect_BayesEst.Detect_BayesEst import Detect_BayesEst
    ```

## Uniform Interface
* DetectBayesEst(constellation)<br>
`@constellation`: a 1-D vector of the constellation map<br>
`@min_var`:***(optional)*** a scalar of the minimal variance (defaut at 2.22e-16)
```python
bayes_est_detector = Detect_BayesEst(constellation);
```
* detect<br>
`@mean`: a vector of the observation of received signals (at least 1D for python, only the last dimension is data)<br>
`@var`: a scalar or vector of the variance of y (at least 1D for python, only the last dimension is data)<br>
`@decoding`: ***(optional)*** a scalar set to false in default. If it is true, we just output the most possible x

***Matlab***
```matlab
[pxyMean, pxyVar] = bayes_est_detector.detect(mean, var);
pxyMean = bayes_est_detector.detect(mean, var, "decoding", true);
```
***Python***
```python
pxyMean, pxyVar = bayes_est_detector.detect(mean, var);
pxyMean = bayes_est_detector.detect(mean, var, decoding=True);
```
