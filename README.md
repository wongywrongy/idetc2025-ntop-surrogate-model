# idetc2025-ntop-surrogate-model

## Problem Overview

This submission addresses Problem Statement 3: Surrogate Modeling for Inverse Design, hosted by nTop at the IDETC 2025 Hackathon.

When optimizing a complex part like a heat exchanger, each design iteration incurs high computational cost due to geometry generation and simulation. To accelerate this process, we aim to build a surrogate physics model that predicts engineering performance characteristics — pressure drop, core surface area, and mass — based on input lattice cell sizes in the X and Y/Z directions.

Once trained, this model is used for inverse design, specifying optimal cell sizes that:

* Minimize pressure drop and part mass
* Maximize surface area

The final performance is evaluated on both accuracy and inference speed. The objective function used for optimization is:

```
Performance = Surface Area - (Mass * 10) - (Pressure Drop * 1500)
```

## Approach Summary

* Trained a multi-output regression model using the provided heat exchanger simulation dataset.
* Inputs:

  * Cell Size X (float)
  * Cell Size Y/Z (float)
* Outputs:

  * Pressure Drop (Pa)
  * Surface Area (mm²)
  * Mass (kg)
* Used the trained model for inverse design via black-box optimization to identify optimal cell sizes.

## Architecture and Workflow

Pipeline:

1. Data preprocessing and normalization
2. Model training and evaluation
3. Inference timing and performance analysis
4. Inverse design via scoring function maximization

Directory layout:

```
├── src/                   # Source code
│   ├── model.py           # Surrogate model
│   ├── optimizer.py       # Inverse design logic
│   └── evaluate.py        # Evaluation script
├── notebooks/             # Development notebooks
├── data/                  # Provided dataset
├── results/               # Output results and metrics
├── slides/                # Presentation files
└── README.md              # Documentation
```

## Results

### Surrogate Model Accuracy

| Output        | RMSE     |
| ------------- | -------- |
| Pressure Drop | XX.X Pa  |
| Surface Area  | XXXX mm² |
| Mass          | X.XX kg  |

* Average inference time per prediction: X.XX ms

### Inverse Design Output

| Cell Size X | Cell Size YZ | Pressure Drop | Surface Area | Mass    | Score  |
| ----------- | ------------ | ------------- | ------------ | ------- | ------ |
| 11.23 mm    | 9.75 mm      | 85.4 Pa       | 13240.0 mm²  | 0.95 kg | 5243.7 |

## Installation and Setup

```bash
git clone https://github.com/[your-org]/idetc2025-ntop-surrogate-model.git
cd idetc2025-ntop-surrogate-model
pip install -r requirements.txt
```

## How to Run

```bash
# Train the model
python src/model.py --train

# Evaluate the model
python src/evaluate.py

# Run inverse design optimization
python src/optimizer.py
```

## Submission Contents

* `models/`: Saved model weights
* `results/`: Final predictions and inverse design outputs
* `slides/final_presentation.pdf`: Summary and diagrams
* `notebooks/`: End-to-end workflow in Jupyter

## Limitations and Future Work

* Currently limited to two design variables; future work could incorporate more design freedom.
* Optimization could be improved using probabilistic surrogate models with uncertainty estimates (e.g., GPR).
* Future extensions may integrate automated geometry generation and evaluation via nTop Automate.
