# Bitcoin Price Predictor
**Author:** Dylan Mendonca and Daniel Gershanik

**Date:** 22-Dec-2020

## Description

This repository contains the code to train, validate, and evaluate a Bitcoin price prediction model. The architecture used is a LSTM. The model looks back a certain number of days and predicts a price on the next day. 

## Folder Structure
Here's a list of files/folders in the directory:
- `core`: Contains the dataloaders, models, and utilities
- `config.json`: The configuration file to adjust hyperparameters, the dataset, etc.
- `main.py`: The controller that puts everything together to run the train-val-test and evaluate workflow
- `run.cmd/run.sh`: Files that are used to run the workflow
- `data`: Contains the data used in the project

## Running a Workflow
To run the workflow and see the results with the data provided in the `data` folder, follow the instructions below:
1) Clone the repo

```bash
git clone 
```
2) Navigate into the app folder

```bash
cd 
```

3) Install the requirements and activate the pipenv shell

```bash
pipenv install
pipenv shell
```
4) Once the shell is activated, run the app from a bash terminal using:
```bash
sh run.sh
```
or from a command line terminal using:
```
run.cmd
```
5) The entire workflow should run using the configuration variables in the `config.json` file. You should see plots pop up. You should close these plots to continue through the training and validation of the workflow
