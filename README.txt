Temporal anomaly detection: calibrating the surprise

README for the TDA database 


The TDA dataset is released in the file TDADateSet.mat 
It is a Matlab file, which can be loaded into matlab using 'load'.

The file includes the following Matlab variables:

- mat {1x1488 cell}: a cell array of binary access matrices of size 4702x11654. 
Each access matrix records database accesses that occurred during a single hour. 
mat{t}(i,j) == 1 if if user i accessed database table j during time interval t. 

- mat_timestamps {1 x 1488 cell}: the timestamps representing the start time 
of each of the time intervals recorded in the access matrices. 
The timestamp format is yyyymmddhhMMss.
The matrices correspond to consecutive hour-long time intervals 
recorded between July 1st, 2015 and August 31st, 2015. 
