""" 
    A general tool for converting data from the
    dictionary format to an (n x k) python list that's 
    ready for training an sklearn algorithm.

    n--no. of key-value pairs in dictonary
    k--no. of features being extracted

    dictionary keys are names of persons in dataset
    dictionary values are dictionaries, 
    where each key-value pair in the dict is the name of a feature,
    and its value for that person.

    In addition to converting a dictionary to a numpy array, 
    the script will separate the labels from the features.
    This is what targetFeatureSplit is for.

    So, to have the poi label as the target,
    and the features of the person's salary and bonus, 
    here's what you would do:

    feature_list = ["poi", "salary", "bonus"] 
    data_array = featureFormat(data_dictionary, feature_list)
    label, features = targetFeatureSplit(data_array)

    The line above (targetFeatureSplit) assumes that the
    label is the _first_ item in feature_list.
    Very importantthat poi is listed first!
"""

## Imports
import numpy as np

## Functions
def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    """
    Convert dictionary to numpy array of features:
    remove_NaN = True will convert "NaN" string to 0.0
    remove_all_zeroes = True will omit any data points for which all the features you seek are 0.0
    remove_any_zeroes = True will omit any data points for which any of the features you seek are 0.0
    sort_keys = True sorts keys by alphabetical order. 

    NOTE: first feature is assumed to be 'poi' and is not checked for removal for zero or missing values.
    """
    return_list = []

    ### Python 3 Compatibility
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print("error: key ", feature, " not present")
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        #### Logic for deciding whether or not to add the data point.
        append = True
        
        #### Exclude 'poi' class as criteria.
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        
        #### Remove data points that are all zero
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        
        #### Remove data points with any zeroes
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        
        #### Append the data point if flagged for addition.
        if append:
            return_list.append( np.array(tmp_list) )

    return np.array(return_list)

def targetFeatureSplit( data ):
    """ 
    Given a numpy array like the one returned from featureFormat, 
    separate out the first feature and put it into its own list 

    Return targets and features as separate lists

    (sklearn can generally handle both lists and numpy arrays as input formats when training/predicting)
    """
    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    return target, features
