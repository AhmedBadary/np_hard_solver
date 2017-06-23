# np_hard_solver
I have implemented 3 different methods, attempting to solve the np-hard problem; multi-diminsional knapsack.
1. Greedy deterministic:
#### Algorithm:
* Sort all the items in the problem based on different heuristics
* Maintain an “incompatibility list”
* Pick the items in a greedy fashion and skip those that invalidate the “incompatibility list”
* Take the max of the outputted sacks where the stopping criterion is breaking a condition
2. Greed Randomized:
#### Algorithm:
* Start at a random CLASS and insert it in a list. Insert classes greedily if they’re
compatible with the classes in the list
* Run Greedy deterministic on the items with the outputted list of classes
* Repeat
3. Brute-Force Randomized:
Algorithm:
* Randomly pick N-items from the list of all items [N~10-27]
vValidate that the items are compatible with each other; else pick again
* Brute-force by finding the powerset of the items and trying every combination
* Take the max
* Repeat.

### Facts:
* The brute force method works best if there are not much conflict amongst the items (their
classes) due to randomly picking items
* Greedily one can completely beat the scoreboard with some layers of randomization
* Even on a super-computer (~4500 GigaFlops/s) Brute-forcing the whole file is practically
impossible.
* An LP formulation is not close to the optimal solution