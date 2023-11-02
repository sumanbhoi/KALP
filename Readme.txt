Requirements:
Pytorch 1.3.0

1) Put the files prescriptions, diagnoses, labevents files in the folder "data"
2) Run data_processing.py to get processed data.
3) Next obtain patient records and vocabulary set from the output in step 2.
4) Calculate Patient similarity from the patient records using steps outlined in the paper.
5) Download datasets from MEDI, SIDER, and AACC website to build lab-disease and lab-drug interaction graph.
6) Run "train_KALP.py" to train the KALP model using the above-mentioned datasets. Set the Test option as "True" to test the trained model.

