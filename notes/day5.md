# Day 5 notes

## Improving the time complexity

So far system has calculated embeddings every time we run the main script and model libraries functions were called each time. Now **embedding calculation is called once** and vectors are saved in **.pkl files** that are simulating the database. It is much faster traversing the file with certain number of vectors than calculating embeddings over and over again.



## Video processing

i have loaded one video and have processed it frame by frame using OpenCV library. Video loads and system recognizes faces but very slow. I need to make some changes to make it faster. Camera load is not implemented yet. 

