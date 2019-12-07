# Air Guitar with Hand Detection

The code is based on @amarlearning ’s
repository https://github.com/amarlearning/Finger-Detection-and-Tracking.

This program that allows you to play different notes at different volumes by moving your hand in front of the webcam.

## Setup
Install Python 3.7 (not 3.8 or higher) from python.org. The latest release of 3.7 as of writing can be found on: https://www.python.org/downloads/release/python-375/

Note that the installation process for different operating systems will differ. If you're on windows, click "Add Python to PATH"
when you're running the installer.

After installing python, launch a terminal window and cd into the project root.
In the terminal, run "pip3 pip install requirements.txt". This will
Install the libraries needed to run the program.

## Running
In a terminal, go to the folder "Finger Detection and Tracking" and run
"python3 FingerDetection.py".

After the program window shows up, cover your hand with all the squares and
press "z" on the keyboard. This will record the color features of your hand
so the program can detect it. (Press z at and time and it will recalibrate)

Make sure the computer volume is not set too high. The program can play unexpected sounds at unexpected volumes
at unexpected times.