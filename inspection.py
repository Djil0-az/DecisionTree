"""to calculate the label entropy at the root(i.e. the entropy of the labels before any splits) and the error rate(the percent of incorrectly classified instances) of
classifying using a majority vote(picking the label with the most examples). You do not need to look at
the values of any of the attributes to do these calculations, knowing the labels of each example is sufficient.
Entropy should be calculated in bits using log base 2.
"""
import pandas as pd
import numpy as np
import math
import sys
from sklearn.preprocessing import LabelEncoder
from collections import Counter
# inspection.py <input.tsv> <output.txt>

if len(sys.argv) != 3:
    print("Invalid number of arguments please use this: inspection.py <input.tsv> <output.txt> ")
    sys.exit(1)
# read tsv file from sys args
tsv_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_csv(tsv_file, sep='\t')

# since the last coloumn is our class we shall use it
target_column = df.iloc[:, -1]
# Convert categorical values such as democrat and republican to numerical labels using label encoding
le = LabelEncoder()
target_column = le.fit_transform(target_column)


def calc_entropy(column):
    """
    Calculate entropy given a pandas series, list, or numpy array.
    """
    # Compute the counts of each unique value in the column
    counts = np.bincount(column)
    # Divide by the total column length to get a probability
    probabilities = counts / len(column)

    # Initialize the entropy to 0
    entropy = 0
    # Loop through the probabilities, and add each one to the total entropy
    for prob in probabilities:
        if prob > 0:
            # use log from math and set base to 2
            entropy += prob * math.log(prob, 2)

    return -entropy


def calc_error(column):
    '''
    Calculate the percentage of error in a given label 
    '''
    label_counts = Counter(column)
    majority_count = max(label_counts.values())
    total_count = len(column)
    # error = 1 - accuracy
    accuracy = majority_count / total_count
    error_rate = 1 - accuracy
    return error_rate * 100


entropy = calc_entropy(target_column)
error = calc_error(target_column)
with open(output_file, 'w') as f:
    f.write("entropy: {}\n".format(entropy))
    f.write("error: {}\n".format(error))
