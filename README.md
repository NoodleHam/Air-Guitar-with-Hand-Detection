# Air Guitar with Hand Detection

Play air guitar unironically with your fingers (kind of).

The code is based on @amarlearning ’s
repository https://github.com/amarlearning/Finger-Detection-and-Tracking.

This program that allows you to play different notes (technically, sin waves of different frequencies) at different volumes by moving your hand in front of the webcam. Created for fun.

For a demo, see *air-guitar-demo.mkv* under the project directory.

## Setup
Install Python 3from python.org.

Note that the installation process for different operating systems will differ. If you're on windows, click "Add Python to PATH"
when you're running the installer.

After installing python, launch a terminal window and cd into the project root.
In the terminal, run `pip3 install -r requirements.txt`. This will
Install the libraries needed to run the program.

If the pip3 package simpleaudio fails to install (which can happen on Linux machines), try installing the *liabasound2* dependency: `sudo apt install libasound2-dev` and then re-run `pip3 install -r requirements.txt` or simply `pip3 install simpleaudio`

## Running
In a terminal, go to the folder "Finger Detection and Tracking" and run
"python3 FingerDetection.py".

After the program window shows up, cover your hand with all the squares and
press "z" on the keyboard. This will record the color features of your hand
so the program can detect it. (Press z at and time and it will recalibrate)

Make sure the computer volume is not set too high. The program may play unexpected sounds at unexpected volumes
at unexpected times though this is not the intention of the author. 
