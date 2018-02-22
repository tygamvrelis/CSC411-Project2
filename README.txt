Instructions:

See the report for system specifications (i.e. Python version and libraries required)




Instructions for setting up project environment (please read to the end of this file prior to performing any of the steps!):

1. Create a master directory to hold the contents of this project

2. Relative to the master directory, create a directory called "Code". Place all the python files in here 

3. Relative to the master directory, create a folder called "Data".
	2.1. In this directory, place mnist_all.mat and snapshot50.pkl
	2.2 In this directory, create a new folder called Faces.
		2.2.1 Within Faces, place actors.txt, female_faces.txt, and male_faces.txt
		2.2.2 Within Faces, unzip the "faces28.zip", "test227.zip", "training227.zip", and "validation227.zip" files provided. After this, there should be 6 folders in Faces, namely "test set28", "training set28", "validation set28", "test set227", "training set227", "validation set227"



Instructions for running the code:

There are 3 different scripts that can be run:
	1. digits.py -- parts 1 through 6 of the project
	2. faces.py -- parts 8 and 9 of the project
	3. deepfaces.py -- part 10 of the project

Prior to running any of these, make sure that the current working directory (os.getcwd()) is set to the "Code" folder created previously. After this, please run the scripts in the order listed above.