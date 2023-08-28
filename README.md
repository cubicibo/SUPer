# SUPer
SUPer is a tool to convert BDN XML + PNG assets to Blu-ray SUP subtitles.
Unlike any other SUP conversion tools, SUPer analyzes and re-renders the subtitles graphics internally to make full use of the the PGS format (Presentation Graphic Stream). Caption files generated with SUPer can feature softsub karaokes, masking, fades and basic moves and are guaranteed to work nicely on your favorite Blu-ray player.

Two output formats are supported: SUP and PES+MUI. The later is commonly used in authoring softwares like Scenarist BD and DVDLogic suite.

## Usage
SUPer is distributed as stand-alone executables (GUI) or as an installable Python package with gui/cli scripts. SUPer expects as an input a BDN XML file with PNG assets in the same folder. SUPer supports two output formats: SUP and PES+MUI for Scenarist BD projects.

The common usage is the following:
- Generate a BDNXML with PNG assets using [ass2bdnxml](https://github.com/cubicibo/ass2bdnxml) or avs2bdnxml.
- Use SUPer to convert the BDNXML to a Blu-ray SUP; simply load a BDN.XML file, set an output file and have an espresso while the fan spins.

### GUI client
Both the standalone executable and the python script `python3 supergui.py` will display the graphical user interface. The window lets you choose your input BDNXML, the output file name and some options to tune the quality and shape the PGS. The GUI always executes aside of a command-line window giving the conversion progress and some logging information.

- Select the input BDN XML file. The file must resides in the same directory as the PNG assets.
- Select the desired output file and extension using the Windows explorer.
- "Make it SUPer" starts the conversion process. The actual conversion progress is printed in the command line window.

#### GUI options
The GUI features tooltips to explain some of the more obscure options. 
- Compression rate: minimum time margin (in %) between two events to perform an acquisition.
- Acquisition rate: lower values reduces the number of acquisition and leads to longer drought
- Quantization: image quantizer to use. PIL+K-Means is low quality but fast. K-Means and pngquant/libimagequant are better but slower.
- Subsampled BDN.XML: Use a 25 or 29.97 fps BDN.XML and generate the SUP as if it was for 50 or 59.94 fps. Handy for animations.
- Allow normal case object redefinition: This can reduce the number of dropped events. The requirements are quite complex and occurences are very rare.

#### GUI config.ini
The config.ini file can be used to specify the relative or absolute path to a quantizer binary (either pngquant[.exe] or libimagequant[.dll, .so]). If the program is in PATH, the name is sufficient. An external quantizer will offer higher quality than the common Python ones (Pillow or a K-Means clustering) and be, in general, faster.

On UNIX systems, pngquant is fairly easily to get in your PATH via brew, apt-get and so on.

### Command line client
`supercli.py` is essentially the command line equivalent to `supergui.py`.

#### CLI Usage
`python3 supercli.py [PARAMETERS] outputfile`

#### CLI Parameters
```
 -i, --input         Input BDNXML file.
 -c, --compression   Time threshold for acquisitions. [int, 0-100, def: 80]
 -r, --comprate      Decay rate to attain time threshold. [int, 0-100, def: 80]
 -q, --qmode         Image quantization mode. [1: PIL+K-Means on fades, 2: K-Means, 3: PILIQ, def: 1]
 -b, --bt            Target BT matrix [601, 709, 2020, def: 709]
 -s, --subsampled    Flag to indicate BDNXML is subsampled (e.g 29.97 BDNXML for 59.94 output).
 -d, --nodts         Dont compute DTS in stream (not compatible with PES).
 -p, --palette       Always write the full palette (enforced for PES).
 -a, --aheadoftime   Allow ahead of time decoding (not compatible with PES).
 -y, --yes           Overwrite output file if it already exists.
 -v, --version       Print the version and exit.
 -w, --withsup       Write SUP aside of PES+MUI assets
 -l, --allow-normal  Allow normal case object redefinition.
```
The output file extension is used to infer the desired output type (SUP or PES).  

### Python package installation
If you plan to execute SUPer with your own Python environment, you must first install the package:<br/>
`python3 -m pip install SUPer`<br/>
Where SUPer is the actual base directory. You should then be able to execute supergui, supercli, or use the internal libraries in your own Python script by calling `import SUPer` or `from SUPer import [...]`.

## How SUPer works
SUPer implements a conversion engine that uses the entirety of the PG specs described in the two patents US8638861B2 and US20090185789A1. PG decoders, while designed to be as cheap as possible, feature a few nifty capabilities that includes palette updates, object redefinition, object cropping and events buffering.

SUPer analyzes each input images and encodes a sequence of similar images together into a single presentation graphic (bitmap). This PG object has the animation encoded in it and a sequence of palette updates will display the sequence of images. This dramatically reduces the decoding and composition bandwidth and allows for complex animations to be performed while the hardware PG decoder is busy decoding the next PG objects.

### PGS Limitations to keep in mind
- There are only two PGS objects on screen at a time. SUPer puts as many subtitles lines as it can to a single PGS object and minimizes the windows areas in which the said objects are displayed. Palette updates are then used to eventually display/undisplay specific lines associated to a given object.
- A hardware PG decoder has a limited bandwidth and can refresh an object ever so often. SUPer distributes the object definitions in the stream and uses double buffering to ease the work of the decoder. However, the bigger the objects (= windows), the longer they will take to decode. SUPer may be obligated to drop events every now and then if an event can't be decoded and displayed in due time. This will happen frequently if the graphics differ excessively between successive events.
- Moves, within a reasonable area, are doable at lower framerates like 23.976, 24 or 25. The ability to perform moves lowers if the epoch is complex or if the PG windows within which the object is displayed are large.

## Special thanks
- TheScorpius666, Masstock, NLScavenger, Prince 7
- FFmpeg libavcodec pgssubdec authors