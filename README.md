# TCRkin

TCRkin is a Python-based tool for fitting kinetic models to experimental data related to TCR (T cell receptor) signaling. It enables parameter estimation using global optimization (DIRECT) followed by local refinement (Gradient Descent), designed to calibrate models against time-course measurements for multiple peptide ligands at varying concentrations.

## Requirements

Tested with Python 3.12 (via Anaconda).

### Install the required dependencies:

command line:

`pip install numpy==1.26.0 matplotlib==3.9.2 scipy==1.13.1 numdifftools==0.9.41`

## Running the Script

To run the main fitting workflow:

command line: 

`python fitting_10_tkinter_8pep_zero_k_minus.py`

## Workflow Steps

1. Load File – Load the .csv file containing your experimental data.
2. Set Parameters – Configure initial values and bounds for model parameters.
3. Run DIRECT – Perform global optimization with the DIRECT algorithm.
4. Run Gradient – Refine the fit using local gradient-based optimization.

## Input File Format

The input should be a .csv file with values separated by semicolons (;).

### Column Structure

Column Description

1. A Time points (in minutes)
2. B–D Peptide A1 (w/o anti-CD8)
3. E–G Peptide A2 (with anti-CD8)
4. H–J Peptide B1 (w/o anti-CD8)
5. K–M Peptide B2 (with anti-CD8)
6. N–P Peptide C1 (w/o anti-CD8)
7. Q–S Peptide C2 (with anti-CD8)
8. T–V Peptide D1 (w/o anti-CD8)
9. W–Y Peptide D2 (with anti-CD8)

Each peptide condition is measured at three distinct concentrations, represented by L1, L2, and L3 (i.e., one column per concentration).

For example, columns B–D contain data for Peptide A at three concentrations in the absence of anti-CD8.

Columns E–G contain the same concentrations for Peptide A in the presence of anti-CD8.

### Feel free to open an issue if you encounter any problems or have suggestions!
