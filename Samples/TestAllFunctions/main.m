clear;
clc;
disp("Testing the result from ");

load("Detect_BayesEst_InOut.mat");

disp("- test the estimation");
bayes_est_detector = Detect_BayesEst(xpool);
[mean, var] = bayes_est_detector.detect(mu_A_B, zigma_A_B, 'decoding', false);
fprintf("  the difference of the mean is %.16f\n", sum(abs(mean - mu_B)));
fprintf("  the difference of the variance is %.16f\n", sum(abs(var - var_B)));

disp("- test the decoding")
[mean, var] = bayes_est_detector.detect(mu_A_B, zigma_A_B, 'decoding', true);
fprintf("  the difference of the mean is %.16f\n", sum(abs(mean - mu_B)) - 12.825280 );