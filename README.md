# TCRkin

### Tested with python 3.12 by anaconda.

### Need to install dependencies:

`pip install numpy==1.26.0 matplotlib==3.9.2 scipy==1.13.1 numdifftools==0.9.41`

### Running script:

`python fitting_10_tkinter_8pep_zero_k_minus.py`

### Steps:

1. Load file.
2. Set up and apply actual params.
3. Run DIRECT.
4. Run Gradient.

### Input file format:

The csv file separated by ";".

1. Column A - time.
2. Columns B-D - peptide A1.
3. Columns E-G - peptide A2.
4. Columns H-J - peptide B1.
5. Columns K-M - peptide B2.
6. Columns N-P - peptide C1.
7. Columns Q-S - peptide C2.
8. Columns T-V - peptide D1.
9. Columns W-Y - peptide D2.

Each 3 columns for peptide contains L1, L2, L3.


