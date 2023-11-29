%% Bayesian Estimation Detector (soft decoding)
% <AUTHOR>
% Xinwei Qu, 22/5/2021
% email: xiqi4237@uni.sydney.edu.au
%
% <INTRODUCTION>
% y represent p(y|x)'s mean, noisePower represents p(y|x)'s variance. Then
% we can assume y fits Gaussian distribution of a mean. Then we test that
% mean of all possible x values. After that, we normalise all Gaussian
% distribution PDF of every x value. (The sum of those must be 1). Then we
% use those Gaussian distribution on every x to recalculate the mean and
% the variance of p(x|y)
classdef Detect_BayesEst < handle
    properties
      constellation {mustBeNumeric}
      min_var {mustBeNumeric} = eps     % the default minimal variance is 2.22e-16
    end
    methods
        % constructor
        % @constellation: the vector including the constellation map
        function self = Detect_BayesEst(constellation, varargin)
            % register all optional inputs
            inPar = inputParser;
            % Set default values                                    
            addParameter(inPar,'min_var', self.min_var, @isnumeric); % Register names
            inPar.KeepUnmatched = true;                             % Allow unmatched cases
            inPar.CaseSensitive = false;                            % Allow capital or small characters
            parse(inPar, varargin{:});                              % Try to load those inputs 
            
            % constellation
            if ~isvector(constellation)
                error("The constellation map must be a vector.");
            else
                self.constellation = constellation;
            end

            % min_var
            self.min_var = inPar.Results.min_var;
        end
        
        % detection
        % <INPUT>
        % @mean:            vector, the observation of received signals
        % @var:             scalar or vector of the variance of y
        % @Decoding:        scalar, if it is decoding, we just output the most possible x 
        function [pxyMean, pxyVar] = detect(self, mean, var, varargin)
            % Inputs Name-Value Pair 
            inPar = inputParser;
            % Set default values
            isDecoding = false;                                     % the default is not decoding                                    
            addParameter(inPar,'Decoding', isDecoding, @islogical); % Register names
            inPar.KeepUnmatched = true;                             % Allow unmatched cases
            inPar.CaseSensitive = false;                            % Allow capital or small characters
            parse(inPar, varargin{:});                              % Try to load those inputs 
            % take inputs
            isDecoding = inPar.Results.Decoding;

            %% Parameter Setting
            % y
            ySize = length(mean);
            mean = mean(:);
            % noisePower
            var = var(:);
            noisePowerSize = length(var);
            if noisePowerSize ~= 1 && noisePowerSize ~= ySize
                error("Noise Power must either a scalar or a vector with the same dimension of y");
            end

            % xPool
            xPoolSize = length(self.constellation);
            xPool_extended = self.constellation(:);
            xPool_extended = xPool_extended.';               % xPool -> a row vector
            xPoolVec = xPool_extended;                       % save the vector form for the last decoding process

            % Estimate P(x|y) using Gaussian distribution
            mean = repmat(mean, 1, xPoolSize);
            xPool_extended = repmat(xPool_extended, ySize, 1);
            pxyPdfExpPower = -1./(2*var).*abs(mean - xPool_extended).^2;
            pxypdfExpNormPower = pxyPdfExpPower - max(pxyPdfExpPower, [], 2);   % make every row the max power is 0
            pxyPdf = exp(pxypdfExpNormPower);
            % Calculate the coefficient of every possible x to make the sum of all possbilities is 1
            pxyPdfCoeff = 1./sum(pxyPdf, 2); 
            pxyPdfCoeff = repmat(pxyPdfCoeff, 1, xPoolSize);                    % make sum a matrix, ySize rows, xPoolSize colums
            % PDF normalisation
            pxyPdfNorm = pxyPdfCoeff.*pxyPdf;

            % recalculate the new mean and variance
            if isDecoding == true
                % decoding
                [~, pxyPdfMaxId] = max(pxyPdfNorm, [], 2);
                pxyMean = xPoolVec(pxyPdfMaxId);
                pxyMean = pxyMean(:);
                pxyVar = nan;
            else
                % no decoding but esitmating
                pxyMean = sum(pxyPdfNorm.*xPool_extended, 2);
                pxyMeanMat = repmat(pxyMean, 1, xPoolSize);
                pxyVar = sum(abs(pxyMeanMat - xPool_extended).^2.*pxyPdfNorm, 2);
                % limit the minimal variance
                pxyVar = max(pxyVar, self.min_var);
            end
        end
    end
end