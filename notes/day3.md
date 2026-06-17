# Day 3 notes



## Testing baseline system 



In terms of biometric tasks, **false accept** and **false reject** are two main metrics used to evaluate a system’s accuracy. False accept represents the number of cases in which the system recognizes a person who is not recorded in the system as someone familiar and grants that person permission to enter. This is the most dangerous case, and we should do everything we can to prevent it. False reject represents the number of cases in which system denies the permission to person who is part of the system. 



Testing this particular baseline system we built there are few conclusions:

\-We should have **as much as possible enrolment data**. False reject percentage went **from 78 to 11** percent just by adding 4 more images.

\-Setting the threshold too high gives guarantees no false accepts but it also can cause increase in false reject cases.

\-Setting the threshold too low can cause great problems and extreme increase of false accept.

\-In this particular system **threshold of 0.65** has been recognized as the best option.

