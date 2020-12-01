# Robust SIMPLS
Robust multivariate regression models using RPCA and SIMPLS.

Traditional linear regression models that leverage the Singular Value Decomposition (SVD) and traditional Principal Component Analysis (PCA) are easily tainted when trained using corrupt data. In our paper entitled "Robust Data-Driven System Identification for Traffic Flow Networks" we mathematically formalize this issue. By leveraging the noise-filtering properties of Robust PCA (RPCA), regression models can be constructed from corrupt data. We also show that is is possible to use "Inductive RPCA" in order to filter new evaluation data at a low computational cost prior to making predictions informed by the proposed model.

The script RobustSIMPLS receives data in the form of excel spreadsheets. Training data informs the model, which is then evaluated using the test data. Both training data and test data should be split into "predictors" and "responses" prior to program execution.

## Robust Data-Driven System Identification for Traffic Flow Networks
M. Clouatre, E. Smith, S. Coogan, M. Thitsa, “Robust Data-Driven System Identification for Traffic Flow Networks,” Transportation Research Record. (Submitted July 2020).


## References
RPCA:

Candès, Emmanuel J., et al. "Robust principal component analysis?." Journal of the ACM (JACM) 58.3 (2011): 1-37.


Inductive RPCA:

Bao, Bing-Kun, et al. "Inductive robust principal component analysis." IEEE Transactions on Image Processing 21.8 (2012): 3794-3800.


The source code pertaining to RPCA is adapted from:

S.L. Brunton and J.N. Kutz, "Sparsity and Compressed Sensing." In: Data-driven  science  and  engineering: Machine  learning,  dynamical  systems,  and  control. Cambridge University Press, 2019. p. 107--109.
