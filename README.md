# Mental-Health-Competition
Code used to train models and run inference for the following competition:
https://www.drivendata.org/competitions/295/cdc-automated-abstraction/page/915/

Training program trains a series of Bert classifiers to take in text and predict the values of binary and mutliclass variables.

Inference program loads all the models and processes the inference data in batches as allowed by memory, outputting the results as a CSV.
