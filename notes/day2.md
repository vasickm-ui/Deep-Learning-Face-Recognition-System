# Day 2 notes



## Implementing baseline system for face recognition

##### 

I implemented a baseline face recognition system using the InsightFace and OpenCV Python libraries. Images of professional tennis players were used for testing.



The model bundle used in this project was **FaceAnalysis buffalo\_l**, which performs face detection and face recognition. During feature extraction, the model converts each detected face into a 512-dimensional embedding vector. Similarity between faces is measured using **cosine similarity**, where a value of 1 indicates identical vectors and a value of -1 indicates vectors pointing in opposite directions. Embedding vector of each face on test image is compared to embedding vectors of images of known people and face with highest score of cosine similarity is choosed as match. There is also **treshold** value of **0.65** that must be exceeded or the result will be "Unknown".



System is far from perfect it makes mistakes on images with bad quality and images with persons making some expressions. One of the solutions would be feeding the model with variations of images not just one "perfect" image. Model runs on **CPU** but it can be modified to run on GPU depending on our needs and computational complexity.

We could also experiment with treshold and with ways of calculating similarities between vectors.

