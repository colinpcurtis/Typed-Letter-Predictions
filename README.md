# Typed-Letter-Predictions
Keras neural network to predict letters from various attributes like pixel length, height, ect.  

Data provided by UCI Machine Learning Repository https://archive.ics.uci.edu/ml/datasets/Letter+Recognition

This is a relatively straightforward implementation of a supervised neural network, but it uses much of the keras functionality and teaches many good practices of machine learning such as transforming raw data into a computer readable format and creating visualizations so we can understand what the network is telling us about the data.  
For example I tested many different combinations of activation functions and hidden layers in the model before I eventually reached what I believe was towards the upper end of what we could expect from accuracy and diminishing marginal returns, which topped out at roughly 91% accuracy on our test dataset. 
