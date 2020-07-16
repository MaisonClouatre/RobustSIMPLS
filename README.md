# Robust SIMPLS
Robust multivariate regression models using RPCA and SIMPLS.

Traditional linear regression models that leverage the Singular Value Decomposition (SVD) and traditional Principal Component Analysis (PCA) are easily tainted when trained using corrupt data. In our paper entitled "Robust Data-Driven System Identification for Traffic Flow Networks" we mathematically formalize this issue. By leveraging the noise-filtering properties of Robust PCA (RPCA), regression models can be constructed from corrupt data.

The script RobustSIMPLS_1 corresponds to the first numerical experiment presented in our paper. This script can receive data in the form of an excel spreadsheet. Prior to model construction, the data is split into training data and test data. Training data informs the model which is then evaluated using the test data.

The script RobustSIMPLS_2 corresponds to the second numerical experiment in which Robust SIMPLS is compared to other prediction methods. This script requires an additional excel spreadsheet as input that contains predictions produced by a neural network for comparison to Robust SIMPLS. In our paper, we the readily available Tensorflow library in Python to construct/ train our neural network and make these predictions.


## References
Source code for RPCA is adapted from:

Brunton SL, Kutz JN. Sparsity and Compressed Sensing. In: Data-driven science and engineering: Machine learning, dynamical systems, and control. New York: Cambridge University Press; 2019. p. 107--109.
