# Day 6 notes

## Video processing

Previously implemented functions work, but slowly. I tried changing the number of frames we process by capturing every n-th frame. When n is smaller than 5, there is virtually no performance improvement. Medium values of n (between 7 and 15) provide good results, but anything larger than that leads to loss of important data. I tried to implement tracking instead of re-detection on every frame but it made no difference.



## Camera stream processing



