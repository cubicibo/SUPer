# SUPer
SUPer is a subtitle rendering and manipulation tool specifically for the PGS (SUP) format. Unlike any other .SUP exporting tools, SUPer re-renders the subtitles graphics internally to make full use of the the BDSup format. Caption files generated with SUPer can feature softsub karaokes, masking and fades and are likely to work nicely on your favorite Blu-Ray player.
 
## Usage
SUPer is made easy to use with two components:
- supergui.py - a graphical user interface to choose your BDNXML, your input and optionally the SUP file to inject.
- example.py - pretty much the same but without any GUI, giving the user more code flexibility.

SUPer has two purposes:
- Calculate and inject optimised subtitle animations in the PGStream.
- Make it easy to edit any element of the PGStream as long as they know some Python scripting. It provides a many functions to let the users manipulate bitmaps, palettes, timing, etc.

## Advices
SUPer expects a blank area before and after any animated subtitle line. So, an epoch may look like this:

<pre>
 ______   ______   ______   ______   ______   ______   ______
| grp2 | | grp2 | | grp2 | | grp2 | |      | | grp3 | | grp3 | 
|      | |      | |      | |      | |      | |      | |      | 
|      | | grp1 | | grp1 | | grp1 | | grp1 | | grp1 | |      |
|______| |______| |______| |______| |______| |______| |______|
</pre>
Where each grpX is optimised as a bitmap object with palette updates. grp2 and grp3 will be assigned to ods_id zero as they overlap spatially (but not in time) during the epoch. grp1 will be assigned to ods_id one.

Conversely, this means:
- SUPer always assume that connected subtitles in area and time will form an animation. If the area of a subtitle is blanked out for a single frame, SUPer assumes this is a new graphics and optimises it independantly.
- If two entirely different subtitles are shown without a gap, those will be optimiser together. This can lead to lowered quality or some gibberish to be on screen.
- SUPer is very greedy about the subtitles areas so the overlapping of two different objects should not happen unless they do overlap visually.
- SUPer always splits the color palette in two when two objects are on screen. I.e Each ODS has 128 possible colors.

### PGS Limitations to keep in mind
- There are only two PGS objects on screen at a time. SUPer puts as many subtitles lines as it can to a single PGS object. Palette updates are then used to eventually display/undisplay specific lines associated to a given object.
- Some hardware PGS decoders have undefined behaviour when performing palette updates with two objects on screen. SUPer takes a conservative approach and redraws all objects, akin to what Scenarist BD does.

## Suggested workflow
Use avs2bdnxml or PunkGraphicStream to generate a .sup file that features all the standard subtitles. Then, with ass2bdnxml or similar, generate a BDNXML for a complex animation (karaoke, partial fades, anything that involves colors). Then, with supergui.py, inject the first .SUP file with the animation and nearby subtitles.

SUPer can also be used to generate the subtitle of an entire media file straight from a BDNXML. The only hard condition is to avoid two unrelated subtitles lines next to each other in time.

## Special thanks
- TheScorpius666
- FFmpeg libavcodec pgssubdec authors
- NLScavenger

## No thanks given to:
- The PGS format authors. This format is a trainwreck and the patent you filled not even correct.
