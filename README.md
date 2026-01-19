# [Biomolecule-AL-Decipher] Deciphering Key Factors of Active LearningPerformance in Biomolecular Design

This repository hosts the official code and datasets for the research paper *Deciphering Key Factors of Active Learning Performance in Biomolecular Design*.

### Core Focus
We systematically investigate the key factors of Active Learning (AL) performance in biomolecular design, and provide a comprehensive benchmark for uncertainty quantification (UQ) algorithms and AL sampling strategies. Our findings reveal the critical factors influencing AL-based fitness optimization efficiency, offering practical guidance for genetic engineering applications.

### Key Outcomes
- Demonstrated the reliability and robustness of ensemble-based UQ algorithms
- Evaluated representative existing AL sampling strategies 
- Identified significant factors determining optimization efficiency (initial settings, distribution sparsity, and sequence similarity)
- Proposed two quantifiable metrics to measure the key influencing factors

### Repository Structure
Below is the detailed directory and file layout of this repository, which helps you quickly locate code, dependency files, and result storage locations. Auto-generated directories (marked below) do not need to be created manually.
biomolecule-al-decipher/
├── checkpoints/ # Stores trained model parameters (auto-generated after training)
├── result/ # Stores experimental results and generated plots (auto-generated after training)
├── src/ # Source code directory
│ ├── AL_sim.py # Training script for AL simulation
│ ├── UQ_DKL.py # Training script for DKL UQ algorithm
│ ├── UQ_Ensemble.py # Training script for ensemble-based UQ algorithm
│ ├── UQ_MCDropout.py # Training script for MC dropout UQ algorithm
│ ├── configs.py # Data-specific configurations
│ ├── evaluation.py # Evaluation and plotting functions
│ ├── metrics.py # Metrics for UQ
│ ├── model_zoo.py # Models for ensemble-based and MC dropout algo
│ ├── model_zoo_dkl.py # The model for DKL algo
│ ├── train.py # Training functions for ensemble-based and MC dropout algo
│ ├── train_DKL.py # Training functions for DKL algo
│ └── utils.py # Other utility functions
├── requirements.txt # Dependencies for ensemble-based & MC dropout UQ experiments
├── requirements_dkl.txt # Dependencies for DKL UQ experiments
└── README.md # Project documentation (this file)

### Quick Start
This is a demonstration. By following this demo, you will be able to plot the calibration curves presented in the paper.
#### 1. Installation
First, clone the repository to your local machine:
```bash
git clone https://github.com/YisuanZ/biomolecule-al-decipher.git
cd biomolecule-al-decipher
```
Then, set up a corresponding virtual environment according to the type of experiment you wish to perform. It should be noted that the separation of the following environments is due to the specific configuration requirements of our laboratory server. For experiments with ensemble-based and MC dropout UQ algorithms:
```bash
# Create and activate Conda virtual environment (named as you like)
conda create -n AL python=3.9.6
conda activate AL
# Install all required dependencies in the environment
pip install -r requirements.txt
```
For experiments with the DKL UQ algorithm:
```bash
# Create and activate Conda virtual environment (named as you like)
conda create -n ALdkl python=3.9.6
conda activate ALdkl
# Install all required dependencies in the environment
pip install -r requirements_dkl.txt
```
You may also combine the environments into one based on your own circumstances.
#### 2. Training and result extraction
Here, we take the ensemble-based UQ algorithm as an example. After ensuring that your environment is consistent with the description in the Installation section, locate the file ./src/UQ_Ensemble.py. This script can be run directly, and you may also configure its function parameters as needed. To execute the script, run the following command:
```bash
python ./src/UQ_Ensemble.py
```
Upon completion of the training process, the model's training parameters will be saved in the `./checkpoints` directory, while the results generated during the experiment will be stored in the `./result` directory.
### 3. Plotting
Once the results have been obtained, you can run the `./src/evaluation.py` script to generate the corresponding plots. It is important to note that the parameters of the function `CaliCurvePlot()` must be consistent with those used during the model training phase. To generate the plots, execute the following command:
```bash
python ./src/evaluation.py
```
Once the script is completed, the plots will be stored in the `./result/calibration` directory.
