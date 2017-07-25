# Project - Detect SMS Spam

Spam detection is one of the major applications of Machine Learning in the interwebs today. Pretty much all of the major email service providers have spam detection systems built in and automatically classify such mail as 'Junk Mail'. 


## Project Overview

In this project, we will be using the [Naive Bayes algorithm](http://scikit-learn.org/stable/modules/naive_bayes.html) to create a model that can classify [a SMS dataset provided by UCI](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) as spam or not spam. Based on the training we give to the model. 

It is important to have some level of intuition as to what a spammy text message might look like. Usually they have words like 'free', 'win', 'winner', 'cash', 'prize' and the like in them as these texts are designed to catch your eye and in some sense tempt you to open them. Also, spam messages tend to have words written in all capitals and also tend to use a lot of exclamation marks. To the recipient, it is usually pretty straightforward to identify a spam text and our objective here is to train a model to do that for us!

Being able to identify spam messages is a binary classification problem as messages are classified as either 'Spam' or 'Not Spam' and nothing else. Also, this is a supervised learning problem, as we will be feeding a labelled dataset into the model, that it can learn from, to make future predictions. 


## Getting Started

### Prerequisites

You'll need to install:

* [Anaconda](https://www.continuum.io/downloads)
* [Python (Minimum 3)](https://www.continuum.io/blog/developer-blog/python-3-support-anaconda)
* [Jupyter Notebook](http://ipython.org/notebook.html)
* [Pandas](https://anaconda.org/anaconda/pandas)
* [Numpy](https://anaconda.org/anaconda/numpy)
* [scikit-learn](https://anaconda.org/anaconda/scikit-learn)
* [Matplotlib](https://anaconda.org/anaconda/matplotlib)

### Data Files

* `SMSSpamCollection` - SMS dataset provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).


## Python Notebook and Scripts

* `Dectect_Spam-Naive_Bayes.ipynb` - Main project file, an IPython Notebook that contains the analysis for the project.

### Opening the Jupyter Notebook

The project `Dectect_Spam-Naive_Bayes.ipynb` can be read using a [Jupyter Notebook](http://ipython.org/notebook.html). There's also an HTML version `Dectect_Spam-Naive_Bayes.html` included for easier viewability.

* Open your Command Prompt (PC) or terminal (Mac or Linux).
* On a PC click the Start button and search for "Command Prompt".
* On a Mac type command + spacebar. Then, type "terminal" in the Spotlight Search. You can also search for "terminal" in finder.
* Navigate to the directory where you downloaded the Jupyter notebook file.
* On a PC you might type: cd C:\Users\username\Downloads\, replacing your username.
* On Mac or Linux you might type: cd ~/Downloads.
* Run the command `jupyter notebook Dectect_Spam-Naive_Bayes.ipynb` in your terminal.

This will open the [Jupyter Notebook](http://ipython.org/notebook.html) in your browser.

#### Special Note

If you try running a code block and get an error message like `no module named matplotlib`, then your distribution of [Anaconda](https://www.continuum.io/downloads) may be missing a package used in the project. That's okay – there's an easy way that you can install these packages. It's as simple as [Googling](https://www.google.com/) the library for easy to use guides on installation!


## Authors

* **[Paul Foley](https://github.com/paulfoley)**
* [Udacity](https://www.udacity.com/)


## License

* <a rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/"> Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>

<a rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/">
	<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" />
</a>


## Acknowledgments

* [UCI Repositories](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
