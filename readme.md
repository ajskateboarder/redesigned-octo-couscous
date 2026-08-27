for linear/quadratic dcts, regardless of activation delta across layers, taking the pullback of some $V_i$ feature always corresponds to feature $U_i$, significantly more than all other features (by 1.0)

exponential dcts aren't like this with many features connected within similar ranges of scores. circuits constructed from this may be vaguely less interpretable as a result (and naively seem to be so far!).

when working with early layers or over a short range, exp dcts tend to produce connections of $U_i \leftarrow V_i$. this effect seems to attenuate with more depth (n=2)

one-to-many feature maps are possible in $U \rightarrow V$ according to feature pullbacks. deembeddings are not meaningful

... istfg i lost the code that showed that "taking the pullback of an output factor, and mean-ablating a combination of top activating input factors" gave you something circuit-like

this entire time i was misunderstanding what DCT ranking was doing, because the codebase doesnt rank output factors! $V$ is the set of input factors! this is annoying