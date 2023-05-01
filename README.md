# SUPer
SUPer is a subtitle rendering and manipulation tool specifically for the PGS (SUP) format. Unlike any other .SUP exporting tools, SUPer re-renders the subtitles graphics internally to make full use of the the BDSup format. Caption files generated with SUPer can feature softsub karaokes, masking and fades and are likely to work nicely on your favorite Blu-Ray player.
 
## Usage
SUPer is made easy to use with two components:
- supergui.py - a graphical user interface to choose your BDNXML, your input and optionally the SUP file to inject.
- example.py - pretty much the same but without any GUI, giving the user more code flexibility.

SUPer has two purposes:
- Calculate and inject optimised subtitle animations in the PGStream.
- Make it easy to edit any element of the PGStream as long as they know some Python scripting. It provides a many functions to let the users manipulate bitmaps, palettes, timing, etc.

SUPer tries to re-use existing object in the stream and exploits the PG decoders capabilites like palette updates to encode animations. This saves bandwidth significantly and enables to perform animations that are otherwise impossible due to hardware limitations of the PG object decoder.

### PGS Limitations to keep in mind
- There are only two PGS objects on screen at a time. SUPer puts as many subtitles lines as it can to a single PGS object and minimizes the said windows (areas where the said objects appears). Palette updates are then used to eventually display/undisplay specific lines associated to a given object.
- A hardware PG decoder has a limited bandwidth and can refresh an object only ever so often. SUPer distributes the object definitions in the binary stream to ease the work of the decoder. SUPer then uses palette updates to link the missing "steps" between two objects definition. However, SUPer defines the steps depending of a similarity measure with the previous bitmaps. If it changes too much, SUPer is obligated to insert the new object in the stream.

## Suggested workflow
- Use SUPer to convert from a BDNXML to a SUP: Simply load a BDNXML file in the GUI, set an output file and have a coffee.
- Use SUPer to inject an existing SUP: Select the SUP to inject as well as the BDNXML file that defines the missing content to inject. You may use ass2bdnxml or avs2bdnxml to generate the BDNXML assets.

## Special thanks
- TheScorpius666
- FFmpeg libavcodec pgssubdec authors
- NLScavenger
- Prince 7
