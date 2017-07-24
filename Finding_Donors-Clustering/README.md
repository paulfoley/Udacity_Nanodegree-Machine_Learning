# Project - Finding Donors for CharityML

This project is designed to show off supervised learning algorithms available in [scikit-learn](https://anaconda.org/anaconda/scikit-learn), as well as show a method for evaluating how each model works. It is important in machine learning to understand exactly when and where a certain algorithm should be used, and when one should be avoided.


## Project Overview

In this project, we will apply supervised learning techniques to help CharityML (a fictitious charity organization) identify people most likely to donate to their cause. 

### Project Steps

* Explore the data to learn how the census data is recorded. 
* Apply a series of transformations and preprocessing techniques to manipulate the data into a workable format. 
* Establish a benchmark for a solution to identifying charity donors.
* Evaluate several supervised learners on the data, and consider which is best suited for the solution. 
* Optimize the model and present it as a solution to CharityML. 
* Explore the chosen model and its predictions, to see how well it's performing.


## Getting Started

### Prerequisites

You'll need to install:

* [Anaconda](https://www.continuum.io/downloads)
* [Python (Minimum 3)](https://www.continuum.io/blog/developer-blog/python-3-support-anaconda)
* [Pandas](https://anaconda.org/anaconda/pandas)
* [Numpy](https://anaconda.org/anaconda/numpy)
* [scikit-learn](https://anaconda.org/anaconda/scikit-learn)
* [Matplotlib](https://anaconda.org/anaconda/matplotlib)

### Data Files

The modified census dataset consists of approximately 32,000 data points, with each datapoint having 13 features. This dataset is a modified version of the dataset published in the paper [*"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",* by Ron Kohavi](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf). The original dataset hosted on [UCI](https://archive.ics.uci.edu/ml/datasets/Census+Income).

#### Features

* `age`: Age
* `workclass`: Working Class (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
* `education_level`: Level of Education (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)
* `education-num`: Number of educational years completed
* `marital-status`: Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
* `occupation`: Work Occupation (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
* `relationship`: Relationship Status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
* `race`: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
* `sex`: Sex (Female, Male)
* `capital-gain`: Monetary Capital Gains
* `capital-loss`: Monetary Capital Losses
* `hours-per-week`: Average Hours Per Week Worked
* `native-country`: Native Country (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)

#### Target Variable

* `income`: Income Class (<=50K, >50K)


## Python Notebook and Scripts

* `Finding_Charity_Donors.ipynb` - Main project file, an IPython Notbook that contains the analysis for the project.

* `visuals.py` - A Python script containing visualization code that is run behind the scenes.

### Opening the Jupyter Notebook

The project `Finding_Charity_Donors.ipynb` can be read using a Jupyter Notebook. There's also an HTML version `Finding_Charity_Donors.html` included for easier viewability.

* Open your Command Prompt (PC) or terminal (Mac or Linux).
* On a PC click the Start button and search for "Command Prompt".
* On a Mac type command + spacebar. Then, type "terminal" in the Spotlight Search. You can also search for "terminal" in finder.
* Navigate to the directory where you downloaded the Jupyter notebook file.
* On a PC you might type: cd C:\Users\username\Downloads\, replacing your username. Learn more about basic terminal commands.
* On Mac or Linux you might type: cd ~/Downloads.
* Run the command `jupyter notebook Finding_Charity_Donors.ipynb` in your terminal.

This will open the iPython Notebook software and project file in your browser.

#### Special Note

If you try running a code block and get an error message like 'no module named matplotlib', then your distribution of Anaconda may be missing a package used in the project. That's okay – there's an easy way that you can install these packages. It's as simple as Googling the library for easy to use guides on installation!


## Authors

* **[Paul Foley](https://github.com/paulfoley)**
* [Udacity](https://www.udacity.com/)


## License

* <a rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/"> Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>

<a rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/">
	<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" />
</a>


## Acknowledgments

* [Ron Kohavi](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf)
* [UCI](https://archive.ics.uci.edu/ml/datasets/Census+Income)
