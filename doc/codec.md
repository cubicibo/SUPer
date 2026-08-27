# Architecture
The encoding engine is designed based on the decoder model constraints. To understand the underlying architecture and design choice, the HDMV PGS decoder model shall be introduced presented.

## Decoder

Here is a scheme of the HDMV PGS decoder model:
```
                                                                                                    Video plane                    
                                                                                                    ─────────────────────┐         
                                                                                                                         │         
                                                                                                                         │         
                                                                                                                         │         
                │                                                                                                        │         
                │ Decoder                                                       ┌───────────────┐                        │         
                │                                                               │ Shadow plane  │                        │         
                │                                                            ┌──┴────────────┐  │                        │         
                │                                                            │ Graphic plane │  │                        │         
┌─────────────┐ │    ┌────────────┐   ┌──────────────┐Rd ┌───────────────┐Rc │               │  │      ┌───────┐         ▼         
│  Transport  ├─────►│  RX Buffer ├──►│   Graphics   │──►│ Object buffer ├──►│               │  │      │       │                Output   
│Packet buffer│ │    │   1  MiB   │   │    Decoder   │   │    4  MiB     │   │               │  │─────►│ 8-bit ├────► Blend  ─────►
└─────────────┘ │    └────────────┘   └───────┬──────┘   └───────▲───────┘   │               │  │      │ CLUT  │       planes      
                │                             │                  │           │               ├──┘      │       │                   
                │                      ┌──────▼───────┐   ┌──────┴──────┐    │               │         └───▲───┘         ▲         
                │                      │ Composition  │   │ Composition │    └───────▲───────┘             │             │         
                │                      │    Buffer    ├──►│ Controller  │────────────┴─────────────────────┘             │         
                                       └──────────────┘   └─────────────┘                                                │         
                                                                                                                         │         
                                                                                                                         │         
                                                                                                    Interactive plane    │         
                                                                                                   ──────────────────────┘         
```

MPEG-TS Transport Packet carry chHDMV Graphics segments are encapsulated in PES packets and carried in one or numerous MPEG-TS Transport Packet. The PES payload of a graphic segment is obtained by combining the MPEG transport packets in the "RX Buffer" (coded data buffer). The Graphics Decoder removes a full segment at once from the RX buffer on its decoding timestamp.
- Object data are processed by the graphics decoder itself. The decoder decodes the bitmap at the pixel output rate Rd of 16e6 pixels per second.
- Palette, composition and overlaying informations are conveyed to the composition buffer, and processed by the composition controller.

The composition controller, based on the composition informations (windows, object references) composes each window onto the shadowed graphic plane. Object data is copied from the buffer at the composition rate Rc of 32e6 pixels per second. Whenever the composition is done, and on the presentation timestamp, the controller swaps the graphic and shadow planes and sets the target palette in the Color Look-up Table (CLUT).

If the current composition merely specifies a palette update, only the CLUT is updated with the specified palette data: the planes are not swapped.

### Definition of a Window
A window is a rectangular area of the graphic plane where objects may be composited to. At most two windows can be defined concurrently.

```
                                                               
      ┌───────────────────────────────────────────────────┐    
      │  Graphic plane                     ┌───────────┐  │    
      │                                    │ window 1  │  │    
      │                                    │           │  │    
      │                                    └───────────┘  │    
      │                                                   │    
      │                                                   │    
      │                                                   │    
      │                                                   │    
      │     ┌───────────────────────────────────────┐     │    
      │     │ window 0                              │     │    
      │     └───────────────────────────────────────┘     │    
      └───────────────────────────────────────────────────┘    
                                                               
```
Presentation graphics can only be displayed within a given window. Any pixel outside of the windows is not addressable. 

### Definition of a Display Set
A display set is a group of graphic segments that defines a single update of the display.

### Definition of an Epoch
An epoch is a continuous set of graphic segments where presentation of different objects data may be seamless, without visible disruption.
Within an epoch, the window(s) shall never change, and the buffer allocations must be managed to never exceed a total of 4 MiB.
An epoch contains one or more display sets.

### Object Buffer
The Object Buffer operates with a sticky allocation scheme. Incoming data specifies a buffer ID and a target size (bitmap shape). Once a slot is allocated, the allocation remains until the next epoch start. Identically sized bitmaps can re-use prior allocations.
Up to 64 uniquely identified allocations may exist.

```
                                                       
  ┌───────────────────────────┌──────────────────────┐ 
  │                           │                      │ 
  │ slot 0, WxH               │ slot 1, AxB          │ 
  │                           │                      │ 
  ┌────────────────────────────────────┌─────────────┐ 
  │                                    │             │ 
  │ slot 9, CxD                        │ slot 2,     │ 
  │                                    │        ExF  │ 
  ┌───────────────────────────┐────────└─────────────┘ 
  │                           │                      │ 
  │ slot 50, WxH              │                      │ 
  │                           │                      │ 
  └───────────────────────────┘                      │ 
  │                                                  │ 
  │              4 MiB object buffer                 │ 
  └──────────────────────────────────────────────────┘ 
```

### RX Buffer (Coded Data Buffer)
A 1 MiB buffer to store incoming PES segments. The sum of segments decoded at a given DTS shall never exceed 1 MiB.

### Graphics Decoder
An independant processing unit that decodes the Run-Length-Encoded bitmaps and store the output directly in the object data buffer. The Graphics decoder outputs the data to the specified memory slot, based on the "object identifier".

### Composition Object
A composition object specifies what to render and how to overlay it onto the graphic plane. It thereby holds references to both the object to display (the object ID), and the target window. The composition object also carries the coordinates of the object.

Up to 2 composition objects can be specified per display update.
- Both composition object can target the same window.
- Both composition object can refer to to the same object in the buffer.
- All composition objects within a display update share the same palette.

### Composition Controller
An independant processing unit that composes a plane, configures the output palette, and swap the visible and working (shadow) planes based on the composition and display informations. The composition controller always composes all of the specified windows.

The Composition Controller operates independently from the Graphics Decoder. However, the Graphics Decoder shall not perform a decoding operation that compromises the copied object referenced by a composition object of an on-going display update.

At most 8 unique display updates can be buffered in advance.

```
                                                                                  ┌─────────────────────────────────────┐  
                                                                                  │             Graphic Plane           │  
                                                                                  │                                     │  
                                 ┌────────────────────────────────┐               │                                     │  
                     Writing     │                                │               │                                     │  
                        ┌───────►│ object 0                       │               │                                     │  
 ┌──────────────┐       │        │                                │               │    ┌───────────────────────┐        │  
 │  Graphics    │       │        ┌──────────────┬─────────────────┤   Copying     │    │ window                │        │  
 │     Decoder  ├───────┘        │              │                 │               │    │                       │        │  
 └──────────────┘                │ object 1     │ object 2     ───┼─────┬─────────┼────┼───►                   │        │  
                                 │              │                 │     │         │    │                       │        │  
                                 ┌──────────────┴─────────────────┤     │         │    │                       │        │  
                                 │                                │     │         │    └───────────────────────┘        │  
                                 │       4 MiB object buffer      │     │         │                                     │  
                                 └────────────────────────────────┘     │         └─────────────────────────────────────┘  
                                                                        │                                                    
                                                             ┌──────────┴─────────┐                                          
                                                             │  Composition       │                                          
                                                             │       Controller   │                                          
                                                             └────────────────────┘                                          
```

### Decoder states
The decoder is entirely controlled by the datastream. The encoder is thereby fully responsible for its proper management. The composition state, carried in each display set, defines what a decoder may do, or be able to do given its state.

#### Reset (Epoch Start)
The decoder is reset every epoch start:
- Both planes are cleared (made fully transparent).
- The object buffer loses all of the prior allocations.
- Palettes are erased.

#### On-going epoch
- Buffer allocations are made whenever a new slot is introduced in the epoch, and these allocations remains.
- The palettes accumulate the defined entries. New palette information only needs to specify the difference.

#### Acquisiton Point
A marker in the stream: a decoder can start to decode the graphic stream from that point onward. The stream shalln't refer to data provided prior to the said Acquisition Point. A decoder is free to flush the palettes and the bitmaps whenever it sees the flag as that data shall no longer be accessed. Buffer allocations are not erased.

#### Normal Case
A standard display update that generally only defines the difference from prior display sets. A decoder looking to catch up on an on-going datastream cannot start decoding from such a display set.

### Decoder timing
The decoder times its operations on the MPEG-TS 90 kHz clock. A single decoding or composition operation, in MPEG-TS 90 kHz ticks, is the ratio between the number of pixels times the said clock, to the respective rate (Rd or Rc). Any fractional remainder is rounded up. As the decoder and composition controller are independent, their operation are parallelized. The total decoding and overlaying time, whenever dealing with two windows, is thereby not always the sum of all durations.

For every incoming display set, the decoder must first clear the display area. This depends on its current state, and the flags in the said display set. The duration to clear the display is denoted $W_d$, and defined as follow:

$$
W_d = \left \lceil{\frac{90000\cdot N_{pixels}}{R_c}}\right \rceil
$$
where $N_{pixels}$ is defined as the number of pixels that requires to be erased.
$$
N_{pixels} = \begin{cases}
 \text{The graphic plane size} &\text{at the Epoch Start}\\
 \sum_{k=0}^{n_{windows}} E(window_k)= \sum_{k=0}^{n_{windows}}\begin{cases}Area(window_{k})&\text{if }window_k\text{ is empty}\\
0 &\text{else}\end{cases}&\text{else}\\
\end{cases}
$$

The decoding time of a display set is driven by the number of pixels to decode, given all of the objects defined therein.

$$
D_{tot} = \sum_{k=0}^{n_{objects}} D_{k} = \sum_{k=0}^{n_{objects}} \left \lceil{\frac{90000\cdot Size(object_{k})}{R_{d}}}\right \rceil
$$

The composition of a window starts whenever:
- Decoding of the display set has been completed.
- Decoding of the object referenced by the composition object targeting that window has been completed.
- Decoding of all objects referenced by all of the composition objects targeting the said window has been completed.

$$
Composition(window_k) = C_i = \begin{cases}
\left \lceil{\frac{90000\cdot Area(window_k)}{R_c}}\right \rceil
&\text{if }window_k\text{ is assigned}\\
0 &\text{else}\\
\end{cases}
$$

The graphics decoder achieves full decoding of a display set prior to the composition controller finishing up the said composition. We can additionally define $DTS_{end}$, the time at which the graphics decoder can starts to decode the subsequent display set data:
$$
DTS_{end} = D_{d} \lt{} PTS
$$

Given all of these, the latest possible decoding time of a display set, denoted DTS, can be derived from its presentation timestamp (PTS) via:

$$
DTS \le{} PTS - decoding\ duration
$$

The decoding duration is affected by the parallelized operation of the graphics decoder and composition controller. That non-linearity appears in the unspecified function $f$ below:

$$
decoding\ duration = max(f(W_d, \sum_{k=0}^{n_{objects}-1}D_{k}), D_{tot}) + C_{last\ referenced\ window}
$$

Thanks to the decoder constraints of a maximum of two composition objects, we can define $f$:

$$
f = max(W_d, D_0) + \begin{cases}C_{0_{i}} &\text{if }D_0\text{ is the only object assigned to }window_i\\
0 &\text{else}\end{cases}
$$

- If both objects are assigned to the same window, $C_{0_{i}}$ is zero, while  $C_{last\ referenced\ window}$ isn't.
- If both objects were decoded in prior display sets and are to be re-used as is, then the decoding duration is merely the sum of the clear and composition time.
- If a display set defines two composition with two objects, decoded a new of one of the object and re-use an existing object from the buffer, then the re-used object shall appear last in the composition list, and the decoding time $D$ associated to that specific object is zero.

For palette-only display update:

$$
DTS \lt{} PTS\text{ , in general: } DTS = PTS - \sum_{k=0}^{n_{windows}} C_{k}
$$

In general, it's recommended to set the decoding duration of a palette-only display update to the time required to compose both windows.

Lastly:
- Some decoding timestamps are shifted by a tick to avoid certain segments having a PTS equal to their DTS, a forbidden case according to the ITU H.222 standard.
- the DTS shall not precede the PTS by more than one second:

$$
PTS - 90000 \lt{} DTS \le{} PTS - decoding\ duration
$$



## Encoder

### Epochs Definitions
SUPer performs a two-pass analysis. The process always tries to identify the smallest possible composition area. The code aggressively searches for epochs, as long epochs have drawbacks.

The first pass is high-level single-threaded and merely uses the XML metadata. The end goal is to identify groups of events close together. The second pass is refined and multi-threaded. Each thread is assigned a group:
- The algorithm combines the alpha layer of every event (in reverse order)
- Every feasible time gap between two events, a brute force search is performed to find the optimial layout.
    - Given the result, a worst case decoding time can be estimated.
    - If the time gap between those two events is sufficient, an epoch is defined.
- The process restarts with the remainder events.
All epochs definitions are ready when all of the groups have been processed by the threads.

At the end of this process, we have a list of epochs with:
- A set of events that makes up the epoch
- One or two windows, which are fixed for the entire epoch (the composition layout)
- Hard minimum decoding and maximum presentation timestamps that may be used by the encoder to arrange the datastream. 

From there on, the encoding process will be epoch to epoch.

### Object detection
To detect continuous display of a graphic, or sensibly similar graphics across numerous display updates, the encoder first stacks the displayed content of each window. Based on two SSIM measure of prior presentations grayscale to the current display, the object detector may decide if the new display is a new object, or a continuation of the current one.

This process is done for all of the windows of the epoch. The output is a minimum list of objects (prospective objects) that should be carried in the datastream to achieve a somewhat sensible output given the input. Each prospective object is associated to a given window.

### Stream shaping
Based on the events, a list of display updates ("DSNode") is created. These nodes gets references to the prospective objects that they should have in display.  A node has sufficient datastream context to provide an approximate, or an exact decoding and presentation timestamps.
To avoid buffer management complexities, SUPer sizes every object displayed within a given window to the maximum rectangle in display in that window within the epoch. Given that fixed sized, the best object placement is determined to minimize the number of forced display refreshes. The rest of the shaping depends on the user settings.
By that point, each node returns exact timestamps:
- Time at which the decoding starts.
- Time at which the decoding ends.
- Time at which the node shall be displayed.

#### Events filtering
Given the exact decoding timestamps (start & end), short-duration nodes that prevents the display of long-standing ones are dropped to achieve a strictly monotonic decoding timeline. Concurrently, the encoder tries to see if a node may comply with the decoding timeline if it is defined as a difference from the prior displays.

To meet the decoding constraints, the encoder may shift the decoding timestamps of a few display set
- Display sets with compressed bitmaps can only be decoded earlier than intended.
- Display sets that merely convey a palette change may have an arbitrary decoding time. It shall only precede the presentation of the palette.

Each node has definitive timestamps, composition state, and objects at the end of that process.

#### Encoding
The bytestream of an epoch is generated in two nested for loop:
- The primary loop generates the Acquisition Point display sets.
- The inner loop generates the Normal Case display sets that follows, those that define the difference from a base display.

While encoding, a dummy decoder with resources is simulated. Each referenced element (buffer slots, palettes) that shall be reserved for a period of time is obtained through that dummy model. If the preceeding steps were done incorrectly, the encoder would run out of resources and the bytestream generation would fail. The encoding step is thereby self-verifying. 