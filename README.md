# SUPer
SUPer is a tool to convert BDN XML + PNG assets to Blu-ray SUP subtitles.
Unlike existing free and professionnal SUP converters, SUPer analyzes and re-renders the caption graphics internally to fully exploit the PGS format. Caption files generated with SUPer can feature softsub karaokes, masking, fades, basic moves, and are guaranteed to work nicely on your favorite Blu-ray player.

Two output formats are supported: SUP and PES+MUI. The later is commonly used in disc authoring softwares like Scenarist or DVDLogic suites.

## Usage
SUPer is distributed as stand-alone executable with a GUI, or as an installable Python package with gui/cli user scripts.

Users who wish to execute SUPer as a Python package with their local Python environment must first install the package:<br/>
`python3 -m pip install SUPer`,  where `SUPer` is the cloned repository folder.

To convert ASSA files to SUP, one must:
- Generate a BDN XML+PNG assets using [ass2bdnxml](https://github.com/cubicibo/ass2bdnxml) or avs2bdnxml.
- Use SUPer to convert the assets to a Blu-ray SUP or a PES+MUI project; load the BDN.XML file, set an output file and format, and optionally have an espresso while the fan spins.

### GUI client
Both the standalone executable and `python3 supergui.py` will display the graphical user interface. The window let one select the input BDNXML, the output file name and tune some options that affect the quality and the stream structure. The GUI always executes aside of a command-line window providing progress and logging information.

- Select the input BDN XML file. The file must resides in the same directory as the PNG assets.
- Select the desired output file and extension using the Windows explorer.
- "Make it SUPer" starts the conversion process. The actual progress is printed in the command line window.

#### GUI config.ini
The config.ini file can be used to specify the relative or absolute path to a quantizer binary (either pngquant[.exe] or libimagequant[.dll, .so]). If the program is in PATH, the name is sufficient. An external quantizer will offer higher quality than the internal one. 

### Command line client
`supercli.py` is essentially the command line equivalent to `supergui.py`.

#### CLI Usage
`python3 supercli.py [PARAMETERS] outputfile`

#### CLI Parameters
```
 -i, --input         Input BDNXML file.
 -c, --compression   Set the time margin required to perform an acquisition, affects stream compression. [int, 0-100, def: 65]
 -a, --acqrate       Set the acquisition rate, lower values will compress the stream but lower quality. [int, 0-100, def: 100]
 -q, --qmode         Set the image quantization mode. [1: PIL+K-Means on fades, 2: K-Means, 3: PILIQ, def: 1]
 -n, --allow-normal  Flag to allow normal case object redefinition, can reduce the number of dropped events on complex animations.
 -t, --threads       Set the number of concurrent threads to use. Default is 1, maximum is 8.
 -b, --bt            Set the target BT matrix [601, 709, 2020, def: 709]
 -p, --palette       Flag to always write the full palette (enforced for PES).
 -y, --yes           Flag to overwrite output file if it already exists.
 -w, --withsup       Flag to write both SUP and PES+MUI files.
 -m, --max-kbps      Set the maximum bitrate to test the output against. Recommended range: [500-16000].
 -e, --extra-acq     Set the min count of palette update after which acquisitions should be inserted [0: off, default: 2]
 -l, --log-to-file   Set (enable) logging to file and set logging level: [10: debug, 20: info, 25: iinf, 30: warnings]
 -v, --version       Print the version and exit.
 --ssim-tol          Adjust the SSIM tolerance threshold. This threshold is used for bitmaps classification in effects [-100; 100, def: 0] 
```
- Image quantization mode 3 ("PILIQ") is either libimagequant or pngquant, whichever specified in config.ini or available in your system.
- The output file extension is used to infer the desired output type (SUP or PES).
- If `--allow-normal`  is used in a Scenarist BD project, one must not "Encode->Build" or "Encode->Rebuild" the PES assets. Scenarist BD does not implement normal case object redefinition and may destroy the stream. However, building or rebuilding are not mendatory to mux the project.
- `--max-kbps` does not shape the stream, it is just a limit to compare it to. Only `--compression` and `--qmode` may be used to reduce the filesize.
- SUPer shall only generate strictly compliant datastream. No option or value will break compliancy.

## Misc
### GUI/CLI options
Here are some additional informations on selected options, especially those that shape the datastream and affect the output:
- Compression rate: minimum time margin (in %) between two events to perform an acquisition.
- Acquisition rate: Additional compression parameter, should be left at default (100 - no compression).
- Quantization: image quantizer to use. PIL+K-Means is low quality but fast. K-Means and pngquant/libimagequant are high quality but slower.
- Allow normal case object redefinition: whenever possible, update a single object out of two. The requirements are complex so this option may have no effect for some files. Also, the palette is split 50/50 in the vicinity of these events in the stream.
- Insert acquisition: Palette effects are encoded in a single bitmap and the output of the last palette update may remain on screen for sufficiently long. Artifacts may come to the viewer attention if they remain visible long enough on the screen. Refreshing the screen will hide potential side effects and improve the perceived quality. You may lower the value if you feel like the filesize increases too significantly, or disable this behavior altogether.

### TL;DR Options
First of all, one should leave the acquisition rate untouched at 100%, unless the stream is highly compressible (i.e includes solely a karaoke).

- No faith in SUPer: Use a low compression rate (< 50%). Import the resulting PES+MUI in Scenarist BD and use the Encode->Rebuild functionality. Scenarist BD will re-write the SUP according to their logic without compromising integrity.
- Have faith in SUPer: Set to allow normal case object redefinition, and use an appropriate compression rate (50-85% typ.). Then, in Scenarist BD, mux your project without ever attempting to Encode->Build/Rebuild the PES+MUI files.

### How SUPer works
SUPer implements a conversion engine that uses the entirety of the PG specs described in the two patents US8638861B2 and US20090185789A1. PG decoders, while designed to be as cheap as possible, feature a few nifty capabilities that includes palette updates, object redefinition, object cropping and events buffering.

SUPer analyzes each input images and encodes a sequence of similar images together into a single presentation graphic (bitmap). This PG object has the animation encoded in it and a sequence of palette updates will display the sequence of images. This dramatically reduces the decoding and composition bandwidth and allows for complex animations to be performed while the hardware PG decoder is busy decoding the next PG objects.

### PGS Limitations to keep in mind
- There are only two PGS objects on screen at a time. SUPer puts as many subtitles lines as it can to a single PGS object and minimizes the windows areas in which the said objects are displayed. Palette updates are then used to eventually display/undisplay specific lines associated to a given object.
- A hardware PG decoder has a limited bandwidth and can refresh the display ever so often. SUPer distributes the object definitions in the stream and uses double buffering to ease the work of the decoder. However, the size of the object sets its decoding time: SUPer may be obligated to drop events every now and then if an event can't be decoded and displayed in due time or can't be encoded as a palette update.
- Moves, within a reasonable area, are doable at lower framerates like 23.976, 24 or 25. The ability to perform moves lowers if the epoch is complex or if the objects are large.

## Special Thanks
- Masstock for advanced testing
- NLScavenger and Prince 7 for samples or testing
- TheScorpius666 for the incredible documentation
- FFmpeg libavcodec pgssubdec authors
