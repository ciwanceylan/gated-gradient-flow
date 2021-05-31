## Learning node representations using stationary flow prediction on large payment and cash transaction networks 
### Ciwan Ceylan, Salla Franzén, Florian T. Pokorny

Banks are required to analyse large transaction datasets as a part of the fight against financial crime. 
Today, this analysis is either performed manually by domain experts or using expensive feature engineering.
Gradient flow analysis allows for basic representation learning as node potentials can be inferred directly from network transaction data.
However, the gradient model has a fundamental limitation: it cannot learn the harmonic component of network flows. 
Furthermore, standard methods for learning the gradient flow are not appropriate for flow signals that span multiple orders of magnitude and contain outliers, i.e.\ transaction data.
In this work, the gradient model is extended to a gated version and we prove that it, unlike the gradient model, is a universal approximator for flows on graphs.
To tackle the mentioned challenges of transaction data, we propose a multi-scale and outlier robust loss function based on the Student-t log-likelihood.
Ethereum transaction data is used for evaluation and the gradient models outperform linear and MLP models using hand-engineered features in terms of relative error.
These results extend to 60 synthetic datasets, with experiments also showing that the gated gradient model learns qualitative information about the underlying synthetic generative flow distributions.

