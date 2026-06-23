# Day 7 notes

## Experimenting with fps arguments

Both the video reader and video writer have an FPS-related parameter. In the video reader, the FPS value determines how often detection is performed. In the video writer, the FPS value determines how many frames are written per second in the output video.



I experimented with these parameters and performed some calculations. So far, the best performance is achieved when detection is performed on every third frame, and the output video is written at 6 frames per second.



## Anti-spoofing check and liveness detection

At this point, the system can recognize faces and correctly identify my face, displaying my name and a similarity score. The problem is that it treats videos and photos of my face as if they were a real person. We need to implement an anti-spoofing check to distinguish between a live person and images or recordings. We are gonna use minifasnet\_v2.onnx model to detect spoof images and liveness. This model is open source. We also need to implement liveness detection that asks user to do some specific movement. 



I also improved image quality function. Now it detects blurry images. 

