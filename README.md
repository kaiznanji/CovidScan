# CovidScan

CovidScan is a 3D convolutional neural network that classifies the presence of COVID-19 in CT scans

## How it works

Dataset: I used the MosMedData dataset to obtain lung CT scans that have signs of COVID-19 and those without. I used an initial sample of this full dataset to test the accuracy of the model I created. The following are the number of CT scans classified into both categories.


### Pre-Processing

I handled image proccessing and manipulation using the SciPy library which is popular for handling 3D images.
