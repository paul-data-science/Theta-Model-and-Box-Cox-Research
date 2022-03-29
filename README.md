# Theta-Model-and-Box-Cox-Research
### ENHANCING THE THETA MODEL AND SEQUENTIAL TIME SERIES PREDICTION TECHNIQUES AND APPLICATIONS TO ECONOMETRIC FACTOR MODELING

This is on going research with NJIT Prof. Steve Taylor. I contributed the Python code for the research.

This research to observe how data transformations can increase Theta forecasting model is still work in progress. So far, I recreated the Theta forecasting model and performances by finding the optimum lambda for the Box Cox transformations.

Notes on the code and outcomes: 
</br>The main function (main.py) calls the functions from the config.py file. It will use the config file to process lambda for each series and also find the optimum lambda that can be used for the entire M4 data set files. It has examples for both Binary Search algorithm and Scipy Opt Min module.
I did not include Multiproc in the code (yet). The Binary Search is the fastest, 7 and 16 mins. The Scipy Opt Min takes longest with 26 and 41 mins. Outputs between Binary Search and Scipy are nearly identical. the Scipy beat the Binary Search in finding the common lambda for all M4 data with improvement score of 1.8% vs 1.7% for the Binary Search.

Outputs were saved on csv files.
