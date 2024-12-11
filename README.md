# SUPer
SUPer is a HDMV PGS ("BD SUP") encoder. It generates fully compliant Presentation Graphic Streams from BDN XML + PNG assets.
Unlike existing free and professionnal SUP converters, SUPer analyzes and re-renders the caption graphics internally to fully exploit the PGS format. Caption files generated with SUPer may feature soft-sub typesetting effects like karaokes, masking, fades, basic moves, and are guaranteed to work on your favorite Blu-ray Disc player.

Two output formats are supported: SUP and PES+MUI. The later is commonly used in professional authoring suites.

## Usage
SUPer is distributed as stand-alone executable, or as an installable Python package with gui/cli user scripts.

If you wish to execute SUPer in your own Python environment, you may first install the package:<br/>
`python3 -m pip install ./SUPer`,  where `SUPer` is the present repository.

## Input file format
SUPer only accepts Sony BDN + PNG assets as input. These may be generated via [ass2bdnxml](https://github.com/cubicibo/ass2bdnxml) or avs2bdnxml. Exports from other tools like SubtitleEdit are untested but should work nonetheless.

## config.ini
The config.ini file contains permanent configuration items. Notably, a relative or absolute path to an image quantizer (either pngquant[.exe] or libimagequant[.dll, .so]) and options can be specified. If pngquant is in PATH, the name is sufficient. An external quantizer will offer higher quality than the internal one.

### Graphical User Interface
Both the standalone executables and `python3 supergui.py` will display the graphical user interface. You can select the input BDN XML, the output file name and adapt the stream structure with some options. The GUI always executes aside of a command-line window providing progress and logging information.

- Select the input BDN XML file. The file must resides in the same directory as the PNG assets.
- Select the desired output file and extension using the Windows explorer.
- "Make it SUPer" starts the conversion process. The actual progress is printed in the command line window.

### Command line interface
`supercli.py` is essentially the command line equivalent to `supergui.py`. 

`python3 supercli.py [PARAMETERS] outputfile`

```
 -i, --input         Input BDNXML file.
 -c, --compression   Set the time margin required to perform an acquisition, affects stream compression. [int, 0-100, def: 65]
 -a, --acqrate       Set the acquisition rate, lower values will compress the stream but lower quality. [int, 0-100, def: 100]
 -q, --qmode         Set the image quantization mode. [0: KD-Means, 1: Pillow, 2: HexTree, 3: libimagequant/pngquant, def: 3]
 -n, --allow-normal  Flag to allow normal case object redefinition, can reduce the number of dropped events on complex animations.
 -k, --prefer-normal Flag to always prefer normal case object redefinition, can reduce the overall bitrate.
 -t, --threads       Set the number of concurrent threads to use. Default is 0 (autoselect), maximum is 8.
 -b, --bt            Set the target BT matrix [601, 709, 2020, def: 709]
 -p, --palette       Flag to always write the full palette (enforced for PES).
 -d, --ahead         Flag to enable palette update buffering to drop fewer events.
 -y, --yes           Flag to overwrite output file if it already exists.
 -w, --withsup       Flag to write both SUP and PES+MUI files.
 -m, --max-kbps      Set the maximum bitrate to test the output against. Recommended range: [500-16000].
 -e, --extra-acq     Set the min count of palette update after which acquisitions should be inserted [0: off, default: 2]
 -l, --log-to-file   Set (enable) logging to file and set logging level: [-10: main report, 10: debug, 20: info, 25: iinf, 30: warnings]
 -v, --version       Print the version and exit.
 --layout            Set the epoch and window definition mode. 2 is the preferred default [2: opportunist, 1: normal, 0: basic].
 --ssim-tol          Adjust the SSIM tolerance threshold. This threshold is used for bitmaps classification in effects [-100; 100, def: 0] 
```

### GUI/CLI options
Here are some additional informations on selected options, especially those that shape the datastream and affect the output:
- Compression rate: minimum time margin (in %) between two events to perform an acquisition.
- Acquisition rate: Secondary compression parameter. Leave it untouched unless a lower bitrate is needed.
- Quantization: [Image quantizer](#image-quantizers) to use. pngquant/libimagequant (3) is recommended.
- Allow normal case: allow to update a single object out of two when there's no enough time to refresh both, it may reduce the count of dropped events.
- Prefer normal case: Update only one composition out of the two, even when decoding time is sufficient to refresh both (which is the default). This can significantly reduce the bitrate, but the composition objects can no longer share palette entries.
- Palette update buffering: decode palette updates early to drop fewer events, right before decoding of new bitmaps.
- Insert acquisition: perform a screen refresh after a palette animation. Some palette animations may cause artifacts that remain on the display, a refresh will hide them. This can impact the bitrate as new bitmaps are inserted in the stream.
- Logging: logging level -10 creates a single file listing solely the filtering decisions, if any (e.g. dropped events, normal case object redefinition).
- Layout mode: steer the epoch definition algorithm way of defining windows. Opportunist is the best and coded buffer-safe, but may be disliked by Scenarist "Build/Rebuild" operation. 1 is an acceptable fall-back. 0 Should never be used.

### Image quantizers
SUPer ships with various internal image quantizers and supports two external ones. The different methods (values for `--qmode`) are enumerated here:

0. **KD-Means**: Slow but high quality quantizer. Should only be used if *PILIQ* (3) is not available and quality is a must.
1. **PIL**: Fast, medium quality. Excellent for low bitrates and files with solely dialogue and karaoke. Glows and gradients will be hideous.
2. **HexTree**: Fast, good quality. Suits nicely files with moderate typesetting effects (no heavy gradients or glows).
3. **PILIQ** (libimagequant / pngquant): High quality, acceptably fast. Recommended default, and it preserves glows and gradients.
    - macOS users must install pngquant via brew, or specify the pngquant executable in config.ini
    - Linux and Windows[^1] users can specify either *libimagequant[.dll, .so]* or pngquant executable in config.ini.

Higher quality quantizers (such as *KD-Means* or *PILIQ*) will generally affect (increase) the bitrate of the output stream.

[^1]: On Windows, compiled binaries ship with libimagequant.dll and the PILIQ library embeds a copy for users of the package in a (virtual) environment.

### Additional tips  
- The output file extension is used to infer the desired output type (SUP or PES).
- `--max-kbps` does not shape the stream, it is simply a limit to compare it to. Only `--compression` and `--qmode` may be used to reduce the filesize.
- If `--allow-normal` (`--prefer-normal`) or `--ahead` is used in a Scenarist BD project, one must not "Encode->Build" or "Encode->Rebuild" the PES assets. Building or rebuilding the PES is not mandatory to mux a project.

### Example
`python3 supercli.py -i ./subtitle01/bdn.xml -c 80 -a 100 --qmode 3 --allow-normal --ahead --palette --bt 709 -m 10000 --threads 6 --withsup ./output/01/out.pes`

### How SUPer works
SUPer implements a conversion engine that makes use of most of the PG specs described in the two patents US8638861B2 and US20090185789A1. PG decoders, while designed to be as cheap as possible, feature a few nifty capabilities like palette updates, object redefinition and events buffering. Notably, SUPer analyzes each input images and encodes a sequence of similar images together into a single presentation graphic (bitmap). This dramatically reduces the decoding and composition bandwidth and allows to perform complex animations while the decoder is decoding the next bitmaps.

- Official PG decoders have a specified bandwidth and can refresh the display ever so often. The size of the largest graphic in an epoch sets the decoding time, and SUPer may be obligated to drop events sporadically to produce a compliant datastream.
- Palette updates are instantaneous and not subject to decoding constraints. SUPer tries to use them whenever possible.

## Special Thanks
- Masstock for advanced testing
- TheScorpius666 for the incredible documentation
- FFmpeg libavcodec pgssubdec authors
