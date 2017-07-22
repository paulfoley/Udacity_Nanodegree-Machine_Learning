# Rescales a Feature to be between 0 and 1
def featureScaling(arr):
	max_value = max(arr)
	min_value = min(arr)

	rescale_values = []
	if max_value == min_value:
		for num in arr:
			rescale_values.append(.5)
	else:
		for num in arr:
			rescale_values.append(float((num - min_value))/float((max_value - min_value)))

	return rescale_values

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print(featureScaling(data))