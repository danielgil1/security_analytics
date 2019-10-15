# COMP90073-Security Analytics Assignment 2



## Machine learning based cyberattack detection

## Daniel Gil <Student Id: 905923>



Note: To quickly see results go to notebook Summary.ipynb and can be seen without running the code. 

## Structure of code
The code is structured as follows:

./  
├── inputs  
├── notebooks  
│   └── tests  
├── outputs  
│   ├── BiFlow  
│   ├── clustering  
│   │   ├── BiFlow  
│   │   └── UniFlow  
│   └── UniFlow  
└── src  

*./:* The root folder contains quick access to Summary.pdf which contains the generation of final results including the attack file and Outliers.pdf which includes a visualizations of attackers and victims per each of the anomaly detection techniques, a zip file that contains the csv file with the attack data and access to SA_Assignment2_Report.pdf.  
*inputs:* Contains Uni-directional and bi-directional flow generated with splunk and python data transformations.  
*notebooks:* Contains the files summarizing clustering techniques, outliers identification and a summary of the attack. Note that there are two notebooks for clustering, one per flow, these two notebooks are the same, just differ in variables configurations to generate each flow features output.  
*outputs:* It has images and csv files with all the outputs from models for each of the flows and clustering techniques used.  
*src:* Contains all the source files used from notebooks. It is a set of helper functions to generate hte ouputs in an organized way.  

If code needs to be run, there are two options:  

## Run the clustering analysis and outliers results from the web.

Quickstart: 
1. Outliers per each of the clustering technique: https://github.com/danielgil1/COMP90073-2/blob/master/notebooks/outliers.ipynb
2. Summary DDoS Attack: Go to the repo https://github.com/danielgil1/COMP90073-2/blob/master/notebooks/Summary.ipynb
 

Go to the URL:  http://115.146.92.184:8888/lab 

Password: 6260

Go to the notebook folder and find the notebooks mentioned above you need to run it.

