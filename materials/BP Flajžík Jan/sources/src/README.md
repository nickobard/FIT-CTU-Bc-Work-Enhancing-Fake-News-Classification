The files provided in this folder contains the following:
- czech_stemmer.py is a script that provides the stemming algorithm for Czech. It is called by the notebook Text_Preprocessing.ipynb with a parameter that defines the aggressivity of the algorithm. In default settings, stemming is set to light.
- glove.6B.100d.txt is a text file that stores pretrained GloVe embedding for English. The data are loaded from the notebook Text_Preprocessing.ipynb and stored in memory.
- vectors_cz_glove_dim300_25.txt is a text file that stores pretrained GloVe embedding for Czech. The data are loaded from the notebook Text_Preprocessing.ipynb and stored in memory.
- ./datasets/ stores the datasets including the data itself and the related media.
- Models.ipynb stores the implementation of the classifiers. The classes are called from the classes stored in Experiments.ipynb.
- Text_Preprocessing.ipynb contains the text preprocessing functions that are directly called from the classes stored in Experiments.ipynb.
- Experiments.ipynb stores classes that define the logic of performing experiments as well as saving the results in the form of latex tables.
- Perform_Experiments.ipynb is the final notebook for performing the experiemnts. It calls the notebook Experiments.ipynb that handles the logic of the program.



To run the experiments you need to do following steps:

1. Run the jupyter notebook in this folder.
2. Open the notebook 'Perform_Experiments.ipynb'. This notebook includes instruction to run the experiments as well as examples of output of the program.  
3. If you want to run the experiments on your data, please create a new subfolder in the 'dataset' directory with the same structure as the provided ones.


