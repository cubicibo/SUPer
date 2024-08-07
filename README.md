# SUPer
SUPer is a tool to convert BDN XML + PNG assets to Blu-ray SUP ("HDMV PGS") subtitles.
Unlike existing free and professionnal SUP converters, SUPer analyzes and re-renders the caption graphics internally to fully exploit the PGS format. Caption files generated with SUPer may feature softsub karaokes, masking, fades, basic moves, and are guaranteed to work nicely on your favorite Blu-ray Disc player.

Two output formats are supported: SUP and PES+MUI. The later is commonly used in professional authoring suites.

## Usage
SUPer is distributed as stand-alone executable with a GUI, or as an installable Python package with gui/cli user scripts.

Users who wish to execute SUPer as a Python package with their local Python environment must first install the package:<br/>
`python3 -m pip install ./SUPer`,  where `SUPer` is the repository folder.

## Input file format
SUPer only accepts Sony BDN + PNG assets as input. Those files may be generated via [ass2bdnxml](https://github.com/cubicibo/ass2bdnxml) or avs2bdnxml. Exports from other tools like SubtitleEdit are untested but should work nonetheless.

### GUI client
Both the released standalone executables and `python3 supergui.py` will display the graphical user interface. The window let one select the input BDN XML, the output file name and tune some options that affect the quality and the stream structure. The GUI always executes aside of a command-line window providing progress and logging information.

- Select the input BDN XML file. The file must resides in the same directory as the PNG assets.
- Select the desired output file and extension using the Windows explorer.
- "Make it SUPer" starts the conversion process. The actual progress is printed in the command line window.

#### GUI config.ini
The config.ini file can be used to specify the relative or absolute path to a quantizer binary (either pngquant[.exe] or libimagequant[.dll, .so]). If pngquant is in PATH, the name is sufficient. An external quantizer will offer higher quality than the internal one. 

### Command line client
`supercli.py` is essentially the command line equivalent to `supergui.py`.

#### CLI Usage
`python3 supercli.py [PARAMETERS] outputfile`

#### CLI Parameters
```
 -i, --input         Input BDNXML file.
 -c, --compression   Set the time margin required to perform an acquisition, affects stream compression. [int, 0-100, def: 65]
 -a, --acqrate       Set the acquisition rate, lower values will compress the stream but lower quality. [int, 0-100, def: 100]
 -q, --qmode         Set the image quantization mode. [1: PIL+K-Means on fades, 2: K-Means, 3: PILIQ, def: 3]
 -n, --allow-normal  Flag to allow normal case object redefinition, can reduce the number of dropped events on complex animations.
 -t, --threads       Set the number of concurrent threads to use. Default is 0 (autoselect), maximum is 8.
 -b, --bt            Set the target BT matrix [601, 709, 2020, def: 709]
 -p, --palette       Flag to always write the full palette (enforced for PES).
 -d, --ahead         Flag to enable palette update buffering to drop fewer events.
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
- If `--allow-normal` or `--ahead` is used in a Scenarist BD project, one must not "Encode->Build" or "Encode->Rebuild" the PES assets. Building or rebuilding the PES is not mendatory to mux a project.
- `--max-kbps` does not shape the stream, it is just a limit to compare it to. Only `--compression` and `--qmode` may be used to reduce the filesize.
- SUPer shall only generate strictly compliant datastream. No option or value will break compliancy.

## Misc
### GUI/CLI options
Here are some additional informations on selected options, especially those that shape the datastream and affect the output:
- Compression rate: minimum time margin (in %) between two events to perform an acquisition.
- Acquisition rate: Secondary compression parameter. Do not change it (100) unless a lower bitrate is needed.
- Quantization: image quantizer to use. pngquant/libimagequant (3) is recommended. PIL (1) outputs lower quality but faster (3). (2) is experimental and should not be used.
- Allow normal case object redefinition: whenever possible, update a single object out of two. May lower the count of dropped events.
- Palette update buffering: buffer palette updates before a new object is decoded to drop fewer events.
- Insert acquisition: perform a screen refresh after a palette animation. Some palette animations may cause artifacts that remain on the display, a refresh will hide them. This can impact the bitrate as new bitmaps are inserted in the stream.

### Example (recommended defaults)
`python3 supercli.py -i ./subtitle01/bdn.xml -c 80 -a 100 --qmode 3 --allow-normal --ahead --palette --bt 709 -m 10000 --threads 6 --withsup ./output/01/out.pes`

### How SUPer works
SUPer implements a conversion engine that uses the entirety of the PG specs described in the two patents US8638861B2 and US20090185789A1. PG decoders, while designed to be as cheap as possible, feature a few nifty capabilities that includes palette updates, object redefinition, object cropping and events buffering.

SUPer analyzes each input images and encodes a sequence of similar images together into a single presentation graphic (bitmap). This PG object has the animation encoded in it and a sequence of palette updates will display the sequence of images. This dramatically reduces the decoding and composition bandwidth and allows for complex animations to be performed while the hardware PG decoder is busy decoding the next PG objects.

### PGS Limitation to keep in mind
- Blu-ray players PG decoders have a limited bandwidth and can refresh the display ever so often. The size of the object sets its decoding time, and SUPer may be obligated to drop events sporadically if it can't be decoded and displayed in due time.
- Palette updates are instantaneous and not subject to decoding constraints. SUPer tries to use them whenever possible.
- Most typesetting effect will work fine, as long as they are within a reasonable area.

## Special Thanks
- Masstock for advanced testing
- NLScavenger and Prince 7 for samples or testing
- TheScorpius666 for the incredible documentation
- FFmpeg libavcodec pgssubdec authors
