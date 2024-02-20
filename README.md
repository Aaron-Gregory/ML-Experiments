# Projects

## Network Pruning

The idea with this project is to search through a neural network and identify neurons that are contributing least to network performance. We can then create a comperable but smaller network by removing those neurons that are not contributing.

We explore the following three pruning methods:
- Magnitude-Based Pruning: This method removes neurons with the smallest average absolute values, assuming that smaller values contribute less to the final output.
- Variance-Based Pruning: This approach removes neurons whose values cluster tightly around some mean.
- Gradient-Based Pruning: This method measures the "impact" of a neuron by multiplying its variance with its average gradient, and removes those with lowest impact first.

**Goals:**
* A video showing ROC curves of a model on MNIST as pruning becomes more aggressive.
* Three curves showing accuracy at 75%, 90%, and 95% precision for varying model sizes.