# Coincidence counts

Imagine we have, for each of $n$ observations:

* A series indicator, indicating which series this observation is part of, and
* A feature indicator, indicating whether or not this observation has
  a particular feature (1 = yes, 0 = no).

This code:

* Identifies unique pairs of observations, and for each pair, whether they are
  in the same series (1) or not (0) (`link_pairs`).
* Identifies observations that *both* have the feature (both equal to 1).

It then splits the pairs into:

* Within a series (sequence matches another): "matching" and
* Not within a series (sequence matches no other): "non-matching".

It then calculates the proportion of *both* values for the feature, within
"matching", and the *both* / "non-matching" proportion.

I (MB) am the copyright holder, and release this code under the 2-clause BSD
license.
