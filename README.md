# Robust SIMPLS
Robust multivariate regression models using RPCA and SIMPLS.

Traditional linear regression models that leverage the Singular Value Decomposition (SVD) and traditional Principal Component Analysis (PCA) are easily tainted when trained using corrupt data. In our paper entitled "Robust Data-Driven System Identification for Traffic Flow Networks" we mathematically formalize this issue. By leveraging the noise-filtering properties of Robust PCA (RPCA), regression models can be constructed from corrupt data.

The script RobustSIMPLS receives data in the form of an excel spreadsheet. Prior to model construction, the data is split into training data and test data. Training data informs the model which is then evaluated using the test data.

## Robust Data-Driven System Identification for Traffic Flow Networks
M. Clouatre,   E. Smith, S. Coogan, M.   Thitsa,   “Robust Data-Driven System Identification for Traffic Flow Networks,” Transportation Research Record. (Submitted July 2020).


## References
Source code for RPCA is adapted from:

Brunton SL, Kutz JN. Sparsity and Compressed Sensing. In: Data-driven science and engineering: Machine learning, dynamical systems, and control. New York: Cambridge University Press; 2019. p. 107--109.
