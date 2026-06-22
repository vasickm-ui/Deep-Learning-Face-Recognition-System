# Day 6 notes

## Video processing

Previously implemented functions work, but slowly. I tried changing the number of frames we process by capturing every n-th frame. When n is smaller than 5, there is virtually no performance improvement. Medium values of n (between 7 and 15) provide good results, but anything larger than that leads to loss of important data. I tried to implement tracking instead of re-detection on every frame but it made no difference.



## Camera stream processing

First step is accessing the camera stream. OpenCV had problems finding a backend that could open the camera device. We needed to set the backend choice to CAP\_ANY.

I saved couple of my personal pictures, calculated average embeddings and added that to vector database files. System is able to recognize my face and other unknown faces.

I tested the camera in various cases. It works fine when I stand far away from the camera and when the image is blurry. However, when I show only half of my face, the camera either detects me as an unknown person or fails to detect any face at all. When I cover my eyes, the camera still detects my face, but the similarity score drops to a value very close to the minimum threshold. When i get out of frame there is no detection at all.

