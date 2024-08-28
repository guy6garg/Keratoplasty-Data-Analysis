# Keratoplasty-Data-Analysis
Keratoplasty surgeries are of 4 types in the dataset: 
**Penetrative Keratoplasty PK, Endothelial Keratoplasty EK, Deep Anterior Lamellar Keratoplasty DALK, and THPK.**

In this assignment, I have worked with the Keratoplasty dataset. (The data is a simulated version of data taken from a hospital system of 500 patients who have undergone various types of Keratoplasty surgeries.)
There are two data files, one with doctor notes for each patient(doctor_notes.csv) and the second with patient records which contain their details like patient ID, gender, age, when they visit last(days since surgery), type of surgery done, and the outcome of the surgery (patient_records.csv). 

The repository has the following deliverables:

1. Exploratory Data Analysis and analysis report. _(assignment_a.ipynb)_
   
2. A model to predict the surgical outcome given patient records with suitable explanation. _(assignment_b.ipynb)_

3. Used the model and created a dashboard using streamlit, which takes in the following input to predict the surgical outcome: _(dashboard.py and assignment_c.ipynb)_
  a. Doctor free text: Free text doctor notes regarding surgery done.
  b. Age: age of the patient
  c. Gender: gender of the patient
  d. Surgery date: The date on which surgery was done

4. A short video of the streamlit dashboard demo. _(streamlit-dashboard-video.webm)_
