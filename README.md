# Machine learning project aiming at detecting and finding the characteristics of the Higgs Boson

**Project 1 of the Machine Learning course at EPFL** - It consists in a Kaggle competition similar to the Higgs Boson Machine Learning Challenge (2014).

**Team members:**
- Jelena Banjac
- Eliott Joulot
- Kevin Pelletier


<br>

>A key property of any particle is how often it decays into other particles. ATLAS is a particle physics experiment taking place at the Large Hadron Collider at CERN that searches for new particles and processes using head-on collisions of protons of extraordinarily high energy. The ATLAS experiment has recently observed a signal of the Higgs boson decaying into two tau particles, but this decay is a small signal buried in background noise.<br>
>`- Kaggle`

<br>

---

Project files:
- `run.py` -> main file that was used to generate the submission
- `toolbox/implementations.py` -> necessary methods are implemented bellow
- `toolbox/*.py` -> generally files we needed to clean data, perform cross validation

Report/Documentation:
- `doc/project_1_report.pdf`

Useful Jupyter Notebooks:
- `notebooks/DataCleaningAndAnalysis.ipynb`
- `notebooks/MethodComparison.ipynb`
- `notebooks/clean_jetnum0.ipynb`
- `notebooks/clean_models_run.ipynb`
- `notebooks/run_model.ipynb`
- `notebooks/visualizee_pandas.ipynb`

Data Profilers:
- `notebooks/train_data_profiling.html`
- `notebooks/test_data_profiling.html`

Submitted file:
- `submission/submission_y_predict.csv` or `submission_y_predict.csv`


---

<br>

As you can see in the notebook called visualize_pandas, the data needs to be cleaned.
There is a lot of -999.000 values. We can split the data using the categorical attribute/column called : PRI_jet_num (can be : 0,1,2,3)
Then by splitting the data we are able to correct the data for each dataset.

Jet num3 : 1477 wrong values : DONE
Jet num2 : 2952 wrong values : in process
Jet num1 : 7562 wrong values + 7 useless columns (all -999.000) : in process
Jet num0 : 26123 wrong values + 10 useless columns (all -999.000) : in process 

To correct these values, apply machine learning using the other parameters to replace the -999.000 by a predicted value. 

Once the data is cleaned, apply ML model on the complete dataset.

Please double check the files in implementation.py / cross-validation.py / build_polynomial.py
We can maybe arrange these files. 
