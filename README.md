# Theta-Model-and-Box-Cox-Research
### ENHANCING THE THETA MODEL AND SEQUENTIAL TIME SERIES PREDICTION TECHNIQUES AND APPLICATIONS TO ECONOMETRIC FACTOR MODELING

(The below summary was quoted from an on going research paper written by NJIT Prof. Steve Taylor. I contributed the Python code for the research.)

"The Theta forecasting model was a notable performer in the M4 timeseries forecasting competition.
In particular, it proved to be the top performing model overall when coupled with a Box-Cox preprocessing
transformation which was unexpected given that many of its competitor models had significantly higher
complexity and degrees of freedom; however, this provided no benefit from a predictive performance per-
spective.
We seek to first extend some of the theoretical and numerical literature related to the Theta model.
Namely, we aim to develop explicit exact or approximate expressions for the q-period forecasting distribution
for this model and related summary statistics. If such forms are intractable to derive, we will consider
corresponding numerical techniques to approximate these distributions.
We note that the preprocessing application of the Box-Cox transform was key to the success of the winning
BC-Theta model in the M4 time series forecasting competition. We seek to consider other potential initial
data transformations to better understand to what extent one may further increase aggregate forecasting
performance over the M4 dataset by considering alternative transforms. We start by optimizing the Box-Cox
transformation and then consider addition inverse CDF based transformation methods. We conduct a broad
search over many common preprocessing transformation techniques to better understand what methods offer
the strongest performance gains."

This research to observe how data transformations can increase Theta forecasting model is still work in progress. So far, I recreated the Theta forecasting model and performances by finding the optimum lambda for the Box Cox transformations.

Notes on the code and outcomes: 
</br>The main function (main.py) calls the functions from the config.py file. It will use the config file to process lambda for each series and also find the optimum lambda that can be used for the entire M4 data set files. It has examples for both Binary Search algorithm and Scipy Opt Min module.
I did not include Multiproc in the code (yet). The Binary Search is the fastest, 7 and 16 mins. The Scipy Opt Min takes longest with 26 and 41 mins. Outputs between Binary Search and Scipy are nearly identical. the Scipy beat the Binary Search in finding the common lambda for all M4 data with improvement score of 1.8% vs 1.7% for the Binary Search.

Outputs were saved on csv files.
