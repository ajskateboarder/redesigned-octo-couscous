for linear/quadratic dcts, regardless of activation delta across layers, taking the pullback of some $V_i$ feature always corresponds to feature $U_i$, significantly more than all other features (by 1.0)

exponential dcts aren't like this with many features connected within similar ranges of scores. circuits constructed from this may be vaguely less interpretable as a result (and naively seem to be so far!).

when working with early layers or over a short range, exp dcts tend to produce connections of $U_i \leftarrow V_i$. this effect seems to attenuate with more depth (n=2)