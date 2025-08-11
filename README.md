Accelerating Design Exploration and Optimization
with Surrogate Physics Models
Files and additional resources can be found at https://learn.ntop.com/asme_hackathon/

Background
When optimizing a complex part, each iteration takes a long time, both due to geometry
generation and simulation compute time. One approach is to sample the design space and use
machine learning to train a surrogate physics model from high-quality simulation data. Unlike
generic black-box ML models, this surrogate is grounded in the parameter space of interest and
reflects the governing physics captured by the simulations. This allows for quicker iteration and
trustworthy inverse design approaches.

Problem Statement
Using the data set containing CFD results and mass properties from a heat exchanger model,
train a surrogate model capable of predicting pressure drop, average flow velocity, core surface
area, and mass given input lattice cell size in the X-direction and Y/Z-direction, and inlet flow
velocity. Once your model is trained, use inverse design to specify the optimal lattice cell sizes
and flow velocity to maximize the surface area while satisfying the specified constraints for
pressure drop, mass, and average flow velocity. The value of a surrogate model is measured in
both accuracy and speed, so you will also be evaluated on how long it takes for your model to
generate a prediction given input parameters.

Generating Additional Data and Testing Model
We have also provided the nTop file of the parameterized heat exchanger design. It is not
required, but you are welcome to generate additional data for training or evaluating your model
using nTop Fluids (you can request a free nTop EDU License here). Your license will include
access to nTop Automate, which allows you to run your nTop notebook through a script
(example Python file also provided). You are not required to generate additional data, but doing
so could help you when training your model for inverse design.

Submission
Your submission should be shared as a zip file or a Public GitHub repo containing the model so
it can be run on the judge’s computer. The judge should be able to input a set of parameters and
receive a prediction for the part's pressure drop, average velocity, surface area, and mass for a
given set of input parameters (Cell Size X, Cell Size Y/Z, Inlet Velocity). Your model should also
be capable of predicting an optimal set of input parameters given the scoring criteria below
using inverse design (input parameters can be specified up to 6 significant figures).
Please also provide a short report detailing your training approach and why you chose the
method you did, along with any relevant visualizations of your model performance. This report
can be the same as the presentation slides or an additional document.

Data Set and Model Evaluation
The data set, evaluation inputs, and optimization results will be constrained to the following
ranges:
● 10mm < Cell Size X < 25mm
● 10mm < Cell Size Y/Z < 25mm
● 2500 mm/s < Inlet Velocity < 3500 mm/s

The data set was generated with a “Cell Size” parameter of 0.4mm for the Flow Analysis block
and will be the same for evaluating your model.
Notes

● The challenge focuses on developing a methodology for training and implementing
surrogate physics models, not on the accuracy of the simulation results.
● The data set was generated using a full factorial experiment design, even though that
might not be the most efficient or effective method for training ML models.

Submissions will be scored in the following categories:

Category Criteria Score/Metrics
Model Prediction
(40%)
How closely does your model predict the
pressure drop of a specified cell size
compared to simulation results in nTop
Fluids, and how long does the prediction
take?
● RMSE of model
prediction compared to
simulation results
● The time it takes for
your model to predict

Inverse Design
(40%)
Determine the optimal lattice cell sizes and
inlet velocities that maximize the surface
area while meeting the following constraints:
● Mass < 125 grams
● Pressure Drop < 8000 Pa
● Avg Velocity > 520 mm/s^2
● Performance of the
parameters prescribed
by your model

Presentation (20%) How well do you describe your approach to
training a surrogate physics model and how
well can your methodology be applied to
come complex design problems?
● Clarity of
documentation and
presentation
● Broader application
and potential impact of
methodology