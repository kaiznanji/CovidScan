# CovidScan

CovidScan is a 3D convolutional neural network that classifies the presence of COVID-19 in CT scans

## How it works

Dataset: I used the MosMedData dataset to obtain lung CT scans that have signs of COVID-19 and those without. The images were classified in the dataset by identifying the percentage of ground glass opacities in the lungs. I used an initial sample of this full dataset to test the accuracy of the model I created. The following are the number of CT scans classified into both categories.


### Pre-Processing

I handled image proccessing and manipulation using the SciPy library which is popular for handling 3D images. The sample dataset used 80% for training and 20% for testing. I also used MatPlotLib to visualize a slice of a CT scan with signs of COVID-19. The image can be viewed below.


## Results



The model was tested and trained using Keras with a Tensorflow backend. I used 10 epochs and a batch size of 2 when compiling the model. The results were not quite promising as we'd hoped achieveing an approximate of 73% for classifying between CT scans with or without traces of COVID-19. This is largely because of the small sample dataset that was initially taken, as well as the resizing of the images while undergoing image proccessing. This resulted in key information and details left out in the training of the model. However, using the full dataset with 30 epochs and a batch size of 2, we obtained an F1-Score of 81%.



## Improvements
The 
