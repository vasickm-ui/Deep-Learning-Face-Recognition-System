# Day 8 notes

## Improving liveness detection

Liveness detection is not working properly. We need to find a way to implement an anti-spoofing feature. There is a strong possibility that camera quality and image format are the main issues.



After experimenting, I decided to use an anti-spoofing model from the public hairymax/AntiSpoofing GitHub repository. This model works partially, and threshold fine-tuning has a significant impact on its performance.

