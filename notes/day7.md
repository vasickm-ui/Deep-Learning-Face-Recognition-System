# Day 7 notes

## Experimenting with fps arguments

Both the video reader and video writer have an FPS-related parameter. In the video reader, the FPS value determines how often detection is performed. In the video writer, the FPS value determines how many frames are written per second in the output video.



I experimented with these parameters and performed some calculations. So far, the best performance is achieved when detection is performed on every third frame, and the output video is written at 6 frames per second.

