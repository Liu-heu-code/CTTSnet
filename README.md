# CTTSnet

Project Structure

- my_dataset.py: data processing
- model.py: model definition
- utils.py: utility functions
- Tsloss.py: Teacher and student training loss
- train.py: Teacher model training
- Student model training.py: Student model training
- Teacher student classification model training.py: Teacher and student classification model training

Data Path Configuration

You need to manually prepare six Excel files, corresponding to the training, validation, and test sets for each of the two classes. The first column of each Excel file should contain the image filenames or relative paths (relative to the image root directory) used to locate the specific images.

Suggested Excel file naming (example):

train_classA.xlsx, train_classB.xlsx

val_classA.xlsx, val_classB.xlsx

test_classA.xlsx, test_classB.xlsx

Image Directory: Place all images in a common root directory (e.g., ./images/). The paths in the Excel files should be relative to this directory.

Configuration:
In the following three Python files, locate the corresponding path variables and update them to your actual paths:

train.py

Student model training.py

Teacher student classification model training.py

Paths to be set include:

Full paths to the six Excel files

Full path to the image root directory
