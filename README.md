# Classifying ECG using Deep Learning

## Installing
Using `pipenv`, it's easy to install all dependencies needed:
```
$ pipenv install
```
You can then start a shell with all required dependencies:
```
$ pipenv shell
```

## The models
There is one convolutional neural network, and one recurrent neural network implemented. These are found in the `cnn-cv.py` and `rnn-cv.py` files respectively. You can run these model with a `train` parameters, like:
```
$ python cnn-cv.py train
```
This will, using cross validation, train and evaluate the model. It will output an average accuracy from all iterations of the cross validation, together with the standard deviation.

If you run the model without any parameter, or by explicitly specifying the `test` parameters, like:
```
$ python cnn-cv.py test
```
The model will be first trained using the training data, and the evaluated against a separate test set.
