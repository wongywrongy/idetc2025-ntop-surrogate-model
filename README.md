# idetc2025-ntop-surrogate-model

Problem Statement 3
nTop

When optimizing a complex part, each iteration takes a long time, both due to geometry generation and simulation compute time. One approach is to sample the design space and use machine learning to train a surrogate physics model from high-quality simulation data. Unlike generic black-box ML models, this surrogate is grounded in the parameter space of interest and reflects the governing physics captured by the simulations. This allows for quicker iteration and trustworthy inverse design approaches.

Using the data set from a heat exchanger, train a surrogate model capable of predicting pressure drop, core surface area, and mass properties given input lattice cell size in the X-direction and Y/Z-direction. Once your model is trained, use inverse design to specify the optimal lattice cell size to minimize pressure drop and mass while maximizing the surface area. The value of a surrogate model is measured in both accuracy and speed, so you will also be evaluated on how long it takes for your model to generate a prediction given input parameters.