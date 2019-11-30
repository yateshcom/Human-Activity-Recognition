 Human-Activity-Recognition

# Human Activity Recognition Using Deep Learning
This project is to build a model that predicts the human activities such as Walking, Walking_Upstairs, Walking_Downstairs, Sitting, Standing or Laying. This dataset is collected from 30 persons(referred as subjects in this dataset), performing different activities with a smartphone to their waists. The data is recorded with the help of sensors (accelerometer and Gyroscope) in that smartphone. This experiment was video recorded to label the data manually. got data from here

# How data was recorded:
By using the sensors(Gyroscope and accelerometer) in a smartphone, they have captured '3-axial linear acceleration'(tAcc-XYZ) from accelerometer and '3-axial angular velocity' (tGyro-XYZ) from Gyroscope with several variations. The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain.

# Attribute Information:
For each record in the dataset it is provided:

Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration.
Triaxial Angular velocity from the gyroscope.
A 561-feature vector with time and frequency domain variables.
Its activity label.(WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)
An identifier of the subject who carried out the experiment.
You can check my Total work in ipynb note book and github link

# Some Analysis findings:
No of Datapoints per Activity
No of Datapoints per Activity
Plotted tBodyAccMag_mean feature find some interesting plot that we can devide activities into stationary and Moving(Dynamic) see below plot
Stationary and Moving activities
Magnitude of an acceleration can saperate it well - See below plot
Magnitude of an acceleration
Tried TSNE plots with different perplexity and No of iterations , got similar results as below Magnitude of an accelerationTSNE
Machine Learning Models:
Data:
We have 561 handcoded features of raw series data.
Raw data signals in each axis for Accelerometer and Gyroscop. also Accelerometer was divided into body and total. i.e body = total - gravitational force.
Respective label information
Data divided into Train and Test
# Models
Tried some machine learning algoritms and tuned hyperparameters. you can check my entire results in above ipynb notebook got best results with Linear SVC with 96.5%. Check Test confusion matrix below. Linear SVC
LSTM to Classify features With Raw series data (SIGNALS ="body_acc_x","body_acc_y","body_acc_z","body_gyro_x","body_gyro_y","body_gyro_z","total_acc_x","total_acc_y","total_acc_z")
Used tensorflow and Keras to build models and Tuned Hyperparameters with Hyperas. Tried 1 and 2 layer LSTM because of lack of hardware to train. Tuend all hyperparameters and got Test accuracy if 91.99%. Below is Test Confusion Matrix. you can check my entire results in above ipynb notebook
LSTM
CNN with 1d Convolution of Raw data
Tried CNN wit 1d Conv and tuned hyperparameters with Hyperas. Got best test accuracy 92.3%. you can check my entire results in above ipynb notebook. below is Test confusion matrix CNN
Divide and Conquer-Based with 1D CNN for Raw series Data
in Data exploration section we observed that we can divide the data into dynamic and static type so divided walking,waling_upstairs,walking_downstairs into category 0 i.e Dynamic, sitting, standing, laying into category 1 i.e. static.
used 2 more classifiers seperatly for classifying classes of dynamic and static activities. so that model can learn differnt features for static and dynamic activities as below
Divide and Conquer-Based with 1D CNN Trained these 3 cnn models , Tuned hyperparmeters and written prediction pipeline. you can check my entire results in above ipynb notebook. Got Test accuracy of 96.9%.. Below is Test confusion Matrix. Divide and Conquer-Based with 1D CNN
