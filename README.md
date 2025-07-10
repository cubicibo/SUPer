# SUPer
SUPer is a HDMV PGS ("BD SUP") encoder. It generates fully compliant Presentation Graphic Streams from BDN XML + PNG assets. Caption files generated with SUPer may feature soft-sub typesetting effects like karaokes, masking, fades, basic moves, and are guaranteed to work on your favorite Blu-ray Disc player.

## Usage
SUPer is distributed as stand-alone executable, or as an installable Python package with gui/cli user scripts.

If you wish to execute SUPer in your own Python environment, you may first install the package: `python3 -m pip install ./SUPer`,  where `SUPer` is the present repository.

## Input and Output formats
- Only Sony BDN + PNG assets are accepted as input. These may be generated via [ass2bdnxml](https://github.com/cubicibo/ass2bdnxml) or avs2bdnxml.
- Output: SUP and/or PES+MUI. The later is commonly used in professional authoring suites.

## config.ini
The config.ini file stores permanent configurations. Notably, a relative or absolute path to an image quantizer (either pngquant[.exe] or libimagequant[.dll, .so]). If pngquant is in PATH, the name is sufficient. An external quantizer may offer higher quality than the internal ones.

### Graphical user interface
Both the standalone executables and `python3 supergui.py` will display the graphical user interface. You can select the input BDN, the output file name and select the encoder features. The GUI always executes aside of a command-line window providing progress and logging information.

- Select the input BDN XML file. It must be in the same directory as the PNG assets.
- Select the desired output file and extension.
- "Make it SUPer" starts the conversion process. The actual progress is printed in the command line window.

### Command line interface
`supercli.py` is the command line equivalent to `supergui.py`.

`python3 supercli.py [PARAMETERS] outputfile`

```
 -i, --input         Input BDNXML file [Mandatory]
 -c, --compression   Set the time margin required to perform an acquisition, affects stream compression. [int, 0-100, def: 65]
 -a, --acqrate       Set the acquisition rate, lower values will compress the stream but lower quality. [int, 0-100, def: 100]
 -q, --quantizer     Set the image quantization mode. [0: Qtzr, 1: Pillow, 2: HexTree, 3: libimagequant/pngquant, def: 3]
 -n, --allow-normal  Flag to allow normal case object redefinition, can reduce the number of dropped events on complex animations.
 -k, --prefer-normal Flag to always prefer normal case object redefinition, can reduce the overall bitrate.
 -t, --threads       Set the number of concurrent threads to use. Default is 0 (autoselect), maximum is 8.
 -b, --bt            Set the target BT matrix [601, 709, 2020, def: 709]
 -p, --palette       Flag to always write the full palette (enforced for PES).
 -d, --ahead         Flag to enable display set buffering to drop fewer events.
 -y, --yes           Flag to overwrite output file if it already exists.
 -w, --withsup       Flag to write both SUP and PES+MUI files.
 -m, --max-kbps      Set the maximum bitrate to test the output against. Recommended range: [500-16000].
 -e, --extra-acq     Set the min count of palette update after which acquisitions should be inserted [0: off, default: 2]
 -l, --log-to-file   Set (enable) logging to file and set logging level: [-10: main report, 10: debug, 20: info, 25: iinf, 30: warnings]
 -v, --version       Print the version and exit.
 --layout            Set the epoch and window definition mode. 2 is the preferred default [2: opportunist, 1: normal, 0: basic].
 --ssim-tol          Adjust the SSIM tolerance threshold. This threshold is used for bitmaps classification in effects [-100; 100, def: 0]
 --redraw-period     Set the period to redraw the screen, useful for streams with long lasting events [1.0+, def: 0 (off)]
```

### Options breakdown
Here is a brief summary of the features that shape the output datastream:
- Compression rate: minimum time threshold (in %) between two events to perform a mandatory refresh over a palette update.
- Acquisition rate: Secondary compression parameter. Leave it untouched unless a lower bitrate is needed.
- Quantization: [Image quantizer](#image-quantizers) to use. pngquant/libimagequant (3) is recommended.
- Allow normal case[^1]: allow to update a single object out of two when there is not enough time to refresh both. It may reduce the count of dropped events.
- Prefer normal case[^1]: Update only one composition out of the two, even when decoding time is sufficient to refresh both. This can significantly reduce the bitrate, but compositions cannot share palette entries in these situations.
- display set buffering[^1]: allow to decode data early to drop fewer events.
- Insert acquisition: perform a screen refresh after a palette animation. Some palette animations may cause artifacts that remain on the display, a refresh will hide them. This can impact the bitrate as new bitmaps are inserted in the stream.
- Logging to file: level -10 creates a single file listing solely the filtering decisions, if any (e.g. dropped events, normal case object redefinition).
- Layout mode: steer the epoch definition algorithm way of defining windows. Opportunist is the best and coded buffer-safe but may be disliked by authoring suites "Build/Rebuild" feature. 1 is an acceptable fall-back. 0 Should never be used.
- Redraw period: Insert periodic screen refreshes on long lasting events that exceed the specified value. Low values may dramatically increase the bitrate.
- Maximum bitrate: merely a bitrate to test the output against. It does not shape the stream, it is merely to help authorers budget the bitrate.

[^1]: If `--allow-normal` (`--prefer-normal`) or `--ahead` is used in a Scenarist BD project, one must not "Encode->Build" or "Encode->Rebuild" the PES assets. Building or rebuilding the PES is not mandatory to mux a project.

### Image quantizers
SUPer ships with various internal image quantizers and supports two external ones. The different methods (values for `--quantizer`) are enumerated here:

0. **Qtzr**: High quality and reasonably fast quantizer, albeit generates the largest file sizes.
1. **Pillow**: Fastest but low quality. Excellent for low bitrates and files with solely dialogue and karaoke. Glows and gradients will be hideous.
2. **HexTree**: Fast and good quality. Suits nicely files with moderate typesetting effects (no heavy gradients + large glows).
3. **PILIQ** (libimagequant / pngquant): delivers the best quality, acceptably fast.
    - macOS users must install pngquant via brew, or specify the pngquant executable in config.ini
    - Linux and Windows[^2] users can specify either *libimagequant[.dll, .so]* or pngquant executable in config.ini.

Higher quality quantizers will generally consume more bandwidth:
- Quality: (highest) *PILIQ > Qtzr > HexTree >> Pillow* (lowest).
- Bandwidth usage: (highest) *Qtzr > PILIQ > HexTree >> Pillow* (lowest).

[^2]: On Windows, compiled binaries ship with libimagequant.dll and the PILIQ library embeds a copy for users of the package in a (virtual) environment.

### Example
`python3 supercli.py -i ./subtitle01/bdn.xml -c 80 -a 100 --quantizer 3 --allow-normal --ahead --palette --bt 709 -m 10000 --threads 6 --withsup ./output/01/out.pes`

### References
- US20090185789A1 (Panasonic) - Stream shaping, decoder model, segments timing and stream compliance
- US8638861B2 (Sony)- Segments syntax and buffering
- US7620297B2 (Panasonic) - Decoder model, references management

### How SUPer works
SUPer implements an encoding engine that exploits most of the Presentation Graphic features presented in the aforementionned patents. These nifty capabilities like palette updates, events buffering and objects management must be taken care of by the encoder, which SUPer does. Notably, it analyzes every input images and encodes sequences of similar images together into a single presentation graphic (bitmap) to reduce bandwidth and enables for complex animations within the strict specifications.

Decoders are specified to operate at a fixed bandwidth and can refresh the display ever so often. The size of the largest graphic in an epoch sets the decoding delay: SUPer may drop events sporadically to produce compliant datastreams.

## Special Thanks
- Masstock for advanced testing
- TheScorpius666 for the incredible documentation
- FFmpeg libavcodec pgssubdec authors
