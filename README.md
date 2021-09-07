# The all-new HI-SPACE

<img src="https://img.shields.io/badge/Raspberry%20Pi-A22846?style=for-the-badge&logo=Raspberry%20Pi&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white"/>
<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen"/>

I wrote a bit about the HI-SPACE table and my motivation for this project [here](https://medium.com/@ahslaughter/the-hi-space-table-4ac7f2d9f26c).

There's also a fun little paper describing the original HI-SPACE (and a fun application-- virtual air hockey!) 
[here](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.99.4319&rep=rep1&type=pdf)

# Overview

How hard is it to build a tabletop display that uses hands as input devices today?  

## Equipment/Tools

* Raspberry Pi 3, Model B
   * External display
   * Wired keyboard/mouse
   * Good power supply! (Get the right one-- you're gonna need it!)
* 8 GB SD card
* Raspberry Pi Camera
* OpenCV
* Video projector (...eventually)

# Step 1: Get the Raspberry Pi up and running

* Image the SD card with the Raspberry Pi OS (I used the basic desktop version)
* Set up the dev environment on the Raspberry Pi: 
    - Install the following: 
        * list coming...
* Set up ```virtualenv``` and get the Python requirements installed. 

I played around for a bit with the camera and OpenCV, but ended up just 
collecting and saving some images and video to use to develop the algorithms off-device.

I also housed the Raspberry Pi + Camera in a small box for some stability. Kinda rough, but 
gets the job done for the time being. 

<img src="https://github.com/ahope/fuzzy-octo-robot/blob/master/resources/box4.png" width="200">

# Step 2: Capture some starting data

**Goal: Capture some still images of the background, and some video of the 
background + a hand reaching into the space.** This will be used in the next 
step to develop the hand tracking algorithms. 

- [ ] TODO: Get the code off the PI and put it in the repo!

# Step 3: Open CV! 

**Goal: Identify different approaches to find and track a hand in the video.**

A few different approaches to try: 

* OpenCV provides a background subtractor-- wonder how far that gets us?
* Try a simple "subtract a photo of the background from the video frame"

# Next Steps

Not there yet, but here's what's coming: 

* Collect more images/video for testing the algorithms
* Profile the algorithms for time/space complexity
* Evaluate whether the algorithms are fast enough, and if not what to do different


