# Symmetric learning for noisy labels

## Running the code

1. Run ```pip install requirements.txt```
2. Create three directories, called: *log*, *model* and *plots*.
3. Run the code by ```python models.py <dataset> <n-epochs> <n-batch> <0|1>```
   1. Parameters: dataset: mnist or fashion_mnist
   2. n-epochs: the number of epochs to run
   3. n-batch: the batch size
   4. 0 if ordinary mode (testing on *MNIST/fashion_MNIST*, creating graphs), 1 if choosing *USPS* as testing set. 