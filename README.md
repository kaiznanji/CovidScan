# CovidScan

CovidScan is a 3D convolutional neural network that classifies the presence of COVID-19 in CT scans

## How it works

Dataset: I used the [MosMedData](https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1) dataset to obtain lung CT scans that have signs of COVID-19 and those without. The images were classified in the dataset by identifying the percentage of ground glass opacities in the lungs. I used an initial sample of this full dataset to test the accuracy of the model I created. The following are the number of CT scans classified into both categories.

<p align="center">
  <img src="https://github.com/kaiznanji/CovidScan/blob/main/images/number_of_scans.png?raw=true" width="300" height="50"/>
</p>

Pre-Processing: I handled image proccessing and manipulation using the SciPy library which is popular for handling 3D images. The image was resized to the dimensions of 64 by 64 by 32 for proccessing and memory purposes. The sample dataset used 80% for training and 20% for testing. I also used MatPlotLib to visualize a slice of a CT scan with signs of COVID-19. The image can be viewed below.

<p align="center">
  <img src="https://github.com/kaiznanji/CovidScan/blob/main/images/img.png?raw=true" width="250" height="200"/>
</p>

## Results

The model was tested and trained using Keras with a Tensorflow backend. I used 10 epochs and a batch size of 2 when compiling the model. The following is the accuracy results of the model trained on the sample dataset: 

<p align="center">
  <img src="https://github.com/kaiznanji/CovidScan/blob/main/images/epochs_10_accuracy.png?raw=true" width="475" height="300"/>
</p>

The results were not quite promising as we'd hoped achieveing an approximate of 73% for classifying between CT scans with or without traces of COVID-19. This is largely because of the small sample dataset that was initially taken, as well as the resizing of the images while undergoing image proccessing. This resulted in key information and details left out in the training of the model. However, using the full dataset with 30 epochs and a batch size of 2, we obtained an F1-Score of 81%.


<p align="center">
  <img src="https://github.com/kaiznanji/CovidScan/blob/main/images/test_accuracy.png?raw=true" width="300" height="30"/>
</p>

## Improvements
For future improvements, I'd like to increase the accuracy score as well as add other related viruses that show presence in lung CT scans. Perhaps our model will be able to differentiate with a high degree of certainty between signs of pnuemonia and COVID-19 in CT scans.
