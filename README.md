# SUPer
SUPer is a subtitle rendering and manipulation tool specifically for the PGS (SUP) format. Unlike any other .SUP exporting tools, SUPer re-renders the subtitles graphics internally to make full use of the the BDSup format. Caption files generated with SUPer can feature softsub karaokes, masking and fades and are likely to work nicely on your favorite Blu-Ray player.
 
## Usage
SUPer is made easy to use with the graphical user interface `supergui.py` - it lets you choose your input BDNXML, the output file name and optionally a SUP file to merge with. A command line client is also available as `supercli.py`. See below for further details.

## Suggested workflow
- Generate a BDNXML with PNG assets using ass2bdnxml, avs2bdnxml or SubtitleEdit.
- Use SUPer to convert the BDNXML to a BD SUP; simply load a BDNXML file in the GUI, set an output file and have an espresso while the fan spins.

## GUI client
This is the client executed when you download the stand-alone binary or when you run `python3 supergui.py`. Its interface is very simple but extensive for all types of conversion.

## Command line client
`supercli.py` is essentially the command line equivalent to `supergui.py`.

### Usage
`python3 supercli.py [PARAMETERS] outputfile`

### Parameters
```
 -i, --input         Input BDNXML file.
 -c, --compression   Time threshold for acquisitions. [int, 0-100, def: 80], 
 -r, --comprate      Decay rate to attain time threshold. [int, 0-100, def: 80], 
 -q, --qmode         Image quantization mode. [0: PIL, 1: PIL+K-Means on fades, 2: K-Means, def: PIL]
 -b, --bt            Target BT matrix [601, 709, 2020, def: 709]
 -n, --ntsc          Flag to force a rescaling of all timestamps by a 1.001 factor.
 -s, --subsampled    Flag to indicate BDNXML is subsampled (e.g 29.97 BDNXML for 59.94 output).
 -f, --softcomp      Set compatibility mode (software decoders don't implement all PGS features). 
 -d, --nodts         Dont compute DTS in stream.
 -y, --yes           Overwrite output file if it already exists.
 -v, --version       Print the version and exit.
```
The output file extension is used to infer the desired output type (SUP or PES).  

## Misc
Some misc and trivia about SUPer, how it works and manages to generate complex stream with animations that work on hardware decoders:

### Behind the scene
SUPer tries to re-use existing object in the stream and exploits the PG decoders capabilites like palette updates to encode animations. This saves bandwidth significantly and enables to perform animations that are otherwise impossible due to hardware limitations of the bandwidth limited PG object decoder.

### PGS Limitations to keep in mind
- There are only two PGS objects on screen at a time. SUPer puts as many subtitles lines as it can to a single PGS object and minimizes the windows areas in which the said objects are displayed. Palette updates are then used to eventually display/undisplay specific lines associated to a given object.
- A hardware PG decoder has a limited bandwidth and can refresh an object only ever so often. SUPer distributes the object definitions in the stream to ease the work of the decoder. SUPer then uses palette updates to link the missing "steps" between two objects definition. However, SUPer defines the steps depending of a similarity measure with the previous bitmaps. If it changes too much, SUPer is obligated to insert the new object in the stream as visual quality remains the most important aspect.


## Special thanks
- TheScorpius666, NLScavenger, Prince 7, Masstock
- FFmpeg libavcodec pgssubdec authors
