> python 3.7.12
> conda 23.11.0
### Hidden Markov Model Guided Amino Acid Mutagenesis

##### Running Python Script:
```
# Set up conda environment
conda env create -f hmm_environment.yml
conda activate hmm

git clone https://github.com/iland24/hmm_guided_mutation
cd hmm_guided_mutation

# Set up parameters in parameters.yml before next cmd
python main.py -p parameters.yml
```

##### Output File System
```
├── LICENSE
├── README.md
├── hmm_environment.yml
├── parameters.yml
├── helper.py
├── main.py
├── data
│   ├── aa_near_sub.txt
│   ├── pairs.txt
│   ├── distance_data.txt
│   ├── indicator_data.txt
│   ├── unique_frames.txt
│   └── final_out.csv
└── figures
    ├── pca_coordinate_data.png
    ├── aic_bic.png
    └── k_mean_inertia.png
```
* Figures related to RMSD indicator data are generated only when HAS_INDICATOR is False.

#### Overview:
This project began with an idea of generating additional information about the enzyme-substrate complex that can be used in choosing beneficial mutation positions, as methods such as directed evolution randomly chooses protein mutation sites.

The aim is to exploit molecular dynamics (MD) data of enzyme-substrate complex and locate amino acid (AA) mutation position in the active site that could increase binding affinity and specificity. Once the positions are located, alternate AA is to be suggested. To acheive this, Hidden Markov Model (HMM) was used to identify optimal positions of the substrate in the active site in the MD simulation and substitutions are proposed based on property and length of the AA. 

**Pipeline Steps:**
1. From MD trajectory data, extract distance measurements between active site AA & substrate atoms.
2. Using the distance data, train Gaussian Emission HMM and label each frame/timepoint of test data with a hidden state.
3. Score each hidden state using indicator data and get emission probabilities (means and standard deviations of features for all states) of the highest scoring hidden state.
4.  Suggest AA substitution based on AA properties using the emission probabilities of the high scoring state.

&nbsp;

#### Input: parameters.yml
Path to parameters.yml is passed on to main.py using the "-p" or "-parameter" flag. Below shows the description of the input parameters in parameters.yml file.
```
# True for verbose output in terminal
VERBOSE: If True, print verbose output, else print minimal progress of the script.

# Trajectory & parameter file name
MD_TRAJECTORY_DATA_PATH: Path to MD trajectory data.
MD_PARAMETER_DATA_PATH: Path to MD parameter.

# Directory names to store outputs
DATA_DIR: Name of directory to store output data.
FIG_DIR: Name of directory to store output figures.

# Feature extraction parameters
SUBSTRATE_NAME: Three-letter substrate name in initial PDB file used for MD simulation.
SUB_AA_DIST: Distance (angstrom) to search for amino acids around substrate.
TRAJ_DOWNSAMPLE_FACTOR: Downsample trajectory data by this factor to search for amino acids near substrate.
VARIANCE_COVERAGE: For choosing percent variance coverage for dimensionality reduction in data preprocessing.

# Indicator data parameters
HAS_INDICATOR:If True, use INDICATOR_PAIRS to get reaction indicator data else do PCA to get rmsd indicator data.
INDICATOR_PAIRS: Atom pairs that can be used to indicate reaction probability (0 atom indexing*). Example: atom 1&2 pair and atom 3&4 pair => [[1,2],[3,4]]
PCA_DOWNSAMPLE_FACTOR: When HAS_INDICATOR is False, PCA is done to calculate rmsd indicator data. MD Frames are downsampled for faster calculation.

# HMM parameters
SAMPLE_TP_LEN: Number of timepoints of each sample.
N_TEST_SAMPLES: Number of test samples. Number of train samples is decided based on this value.
DO_HYPERPARAM_SEARCH: If True, search number of components based on loglikelihood/AIC/BIC, else use HMM_N_COMPONENTS.
HMM_N_COMPONENTS_LS: List of number of components to fit HMM for hyperparameter search. 
HMM_EM_ITERATIONS: Number of EM algorithm iterations when training HMM.
HMM_N_COMPONENTS: Number of components to use when DO_HYPERPARAM_SEARCH is False.
N_TEST_ITERATIONS: Number of iterations for running HMM on test data to find high score frames.
```
&nbsp;

#### Output: final_output.csv
1. Current Residue: original residue at the given residue index of FAP
2. Nearby 3 substrate atoms: three substrate atoms closest to the
current residue
It is in form of python dictionary=> atom index: [atom name,
residue_alphacarbon_to_atom_mean_distance, standard_deviation]
3. AA Suggestion and Length: suggested amino acid and the suggested
amino acid’s length

#### Pipeline Description
HMM is used to model the relationship of the active site and the substrate using distance measurements between them from the simulation. Time series distance data extracted from MD trajectory data is used to train Gaussian Emission HMM. 

Features of the data consists of (a) the distances from the alpha-carbon of AA found near the substrate to each substrate atom and (b) the distance from side-chain hydrogen bond donor/acceptor atoms to substrate hydrogen bond donor/acceptor atoms. Test data is designed to be sampled at an even interval from beginning to end of MD trajectory in order to survery the entire trajectory data. 

Viterbi decoding is then used on the test data to label each time point with a hidden state, grouping the time points based on what normal distributions all the features' values are estimated to have come from at that time point. During catalytic reactions, at the active site, while some AA play role in  holding the substrate for the reaction to begin, other AA stabilizes the transition state of the substrate, directly participating in catalysis. Looking into each feature's distance data distribution of the frames labeled with a favorable state, meaning having a high probability of reaction, it would be possible to recognize active site positions with AA that is either too short or too long. Such AA are potential candidates for substitutions.

<!-- *In test runs of HMM and Viterbi, transition from one state to another in the state sequence generated by Viterbi algorithm did not occur abruptly, leading to extended consecutive frames labeled with same hidden state. -->

Number of states, a hyperparameter of HMM, can either be designated or searched using Akaike Information Criterion (AIC), Bayesian Information Criterion (BIC) and log likelihood of the joint probability of HMM. Based on the writer's experience, log likelihood (LL) steadily rose then plateaued as the number of states was incremented. So, it is advised to search various number of states and view the AIC, BIC and LL graph generated by the pipeline to find the optimal number of states. 

To score each state generated from HMM, indicator data that reflects whether the reaction is more or less likely to occur at each time point is used. If the indicator distance data can be measured from user designated pair or pairs of atoms in the MD trajectory, frames labeled with a state with smallest average distance is filtered out for further downstream analysis. 

If no indicator atom pairs are provided, we assume high probability of catalytic reaction at stable conformations of the enzyme-substrate structure and Root Mean Squre Deviation (RMSD) value showing how close or distant each frame is to the stable conformation is used as the indicator data. To find the stable conformations, Principal Component Analysis (PCA) and clustering is conducted on the trajectory coordinate data. K-means is used to cluster the dimension-reduced data points and the frame closest to the center of the cluster is chosen as the most stable conformation.

Final AA suggestion step is based on the each feature's distance data distribution mentioned prior.

When current residue is polar and
&emsp; if there was hydrogen-bond donor atom included in the 3 nearby substrate atoms
&emsp; &emsp;=> suggest either polar/charged amino acids whose length is within bound 
&emsp;if there was no hydrogen-bond donor atom included in the 3 nearby substrate atom
&emsp;&emsp;=> suggest any amino acids whose length is within bound
         
When current residue is non-polar and
&emsp; if there was hydrogen-bond donor atom included in the 3 nearby substrate atoms
&emsp;&emsp;=> suggest either polar/charged amino acids whose length is within bound
&emsp;if there was no hydrogen-bond donor atom included in the 3 nearby substrate atoms
&emsp;&emsp;=> suggest non-polar amino acids whose length is within bound

* bound = 1 standard deviation from mean

#### Notes
* hmm_guided_mutation script loads MD trajectory data and therefore could take up a lot of RAM.

#### License
* Project is available under the MIT license found in the LICENSE file.