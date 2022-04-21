# SUPer
This library lets the user perform manipulations on PGStream (.SUP) files, used for Blu-Ray captions. It can be used to edit the datastream but also inject new segments, epoch etc.
More notably, SUPer can be used to inject special animation effects abusing palette updates.
 
## Core idea
SUPer has two purposes:
- Make it easy to edit any element of the PGStream as long as they know some Python scripting. It provides a many functions to let the users manipulate bitmaps, palettes, timing, etc.
- Calculate and inject optimised subtitle animations in the PGStream.

Put it simply, use PunkGraphicStream or any similar software to generate a nice PGStream as a .sup file. Then, with ass2bdnxml or similar, generate a BDNXML and bitmaps for a complex animation (karaoke, partial fades, anything that involves colors). SUPer can then be used to load the animation, optimise it and inject it in the previously created PGStream.

PGStream files are insanely complex for nothing, so you may modify the provided example.py file at the moment.

## TODOs
- Fix some bad optimisation choice. Currently this can be done manually by performing some preprocessing on the input files (see Preprocess class).
- Investigate motion with cropping / WDS updates
- Fix a lot of bad typing and probable silly bugs