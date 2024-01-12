Link for pkf files: 
https://drive.google.com/drive/u/0/folders/1kBnk7tuM3b0mleuVYczJGpAv6oGYBz5F

”1 random walk generation.ipynb”: Notebook allowing to generate a random walk of a given
number of step and save important features at each timestep in a csv file in the output file
(csv random walk.csv)
36
• ”2 prediction preprocessing.ipynb”: Notebook allowing to preprocess random walk data to facili-
tate the preparation of linear regressions and predictive models. The preprocessed data obtained
are store as data frame in a pickle structure (preprocessed data w 4000.pkl).
• ”3 prediction.ipynb”: Notebook allowing to fit linear regression models and explore possible
regression using these models. The ”preprocessed data w 4000.pkl” file is loaded in the DropBox
to run this last Notebook with our data.
• ”4 path NMF test.ipynb”: Notebook allowing to launch final simulation and get the plot asso-
ciated to it. In case of SSH run, the main of the class path NMF can be used as it produces the
same plot
• ”5 Lstm position.ipynb”: Notebook used to process the data and train the LSTM networks used
during the final simulation.