#!/bin/bash

# This script converts the shifted-phase data in ECGSignalsDifferentPhase.zip sent by Vasileios
#  to a unique file formatted the same as in /data/ECG5000.txt

cd other

# Decompress files
unzip ECGSignalsDifferentPhase.zip

# Convert them to the same format and into a unique file
for f in ECG5000ShiftedClass*.txt; do { tail -n 1 "$f"; echo; head -n -1 "$f"; } | { tr -s '\r' '\n'; echo; } | { tr -s '\n' ','; echo; } | sed '$ s/.$//' >> ECG5000Shifted.txt; done

# Move the file
mv ECG5000Shifted.txt ../data

# Clean-up
rm ECG5000ShiftedClass*.txt
