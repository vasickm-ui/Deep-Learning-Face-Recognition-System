# Day 4 notes

## Testing baseline system

Today i tested again baseline system in order to determine the optimal parameters. Two main parameters are **threshold** and **number of enroll images** per person.  I used balanced accuracy score defined with given formula **1 - (FRR + 1.5\*FAR)\*2**. This formula punishes bigger FAR ensuring security of system. Having more enroll images eliminates outliers and noise.



## Refactoring code

Everything was stored inside one python script. Functionalities should be separated into their own modules. This way there is no logic repeating and it is much easier to use code base.



## Additional things (research, experimenting and potential ideas)

Using **Laplacian variance** to eliminate blurry images.

I was reading about **other ways to measure optimal performance** of this system (ERR, weight cost, min FAR, min FAR with threshold for FRR).

Other ways to **calculate embeddings** like medoid, clustering, best-quality, weighted average embedding.

