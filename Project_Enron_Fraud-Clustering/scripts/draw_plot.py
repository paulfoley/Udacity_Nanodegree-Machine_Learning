import matplotlib.pyplot as plt

# Draws a KMeans Cluster Scatter Plot
def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    # Some Plotting Code Designed to Help Visualize Clusters

    # Plot Each Cluster With a Different Color
    # Add More Colors for Drawing More than Five Clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    # Place Red Stars Over Points that are POIs
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()