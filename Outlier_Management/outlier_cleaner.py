"""
    Clean away the 10% of points that have the largest residual errors
    (difference between the prediction and the actual net worth).

    Return a list of tuples named cleaned_data where 
    each tuple is of the form (age, net_worth, error).
"""

def outlierCleaner(predictions, ages, net_worths):
    cleaned_data = []
    errors = predictions - net_worths
    for i in range(0, len(predictions)):
        cleaned_data.append((ages[i], net_worths[i], errors[i]**2))
    
    cleaned_data = sorted(cleaned_data, key=lambda data: data[2])
    length = int(len(cleaned_data)- len(cleaned_data)*.1)
    return cleaned_data[:length]

