# RLCore.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://mkschleg.github.io/RLCore.jl/dev)
[![Build Status](https://travis-ci.com/mkschleg/JuliaRL.jl.svg?branch=refactor)](https://travis-ci.com/mkschleg/RLCore.jl)


This is the core to many of my RL frameworks in Julia. It follows some wisdom brought to us by Adam White and Brian Tanner in their development of RLGlue. Some key attributes:

- Options to manage random number generator to ensure reproducibility.
- Hard seperation of Agent, Environment, and Experiment concepts as seperate type trees.
- Minimal api with more complete experiments and ideas in sperate sister repositories.

# Prior Work

The [ReinforcementLearning.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl) package is another really nice project which works to implement core reinforcement learning concepts for Julia. Currently, their package is much more feature complete than this package. While this is the case, the core design principle of the two packages is quite different and I believe the overall goals of the projects are quite different.

### Goals and Principles

  My goal for this project is to provide tools for reinforcement learning researchers to do good research and science. To achieve this goal I've decided on a few core design principles:
  
  1. There should be limited obfuscation between what is written and what runs. A core reason why I decided to do my PhD work in Julia is because of the transparancy of the tools and the absence of object orientation. I believe OOP is a central cause for mistakes in RL and ML empirical studies. Because of this, all functions should be as transparent as possible with minimal layers of composition.
  2. Little or no surprising decisions. While this is less of an issue for this core repository (as there are few decisions I have to make), the overall collection of repositories for reinforcment learning research will limit surprising defaults or undocumented optimizations. These two attributes have become a burden on the community as they often are unreported and hard to find (and impossible if code is not realeased). 
  3. I believe it is the researchers responsibility to make sure their code is consistent. Thus, I often design functions which can use a user managed random number generator (an RNG other than the GLOBAL). This is never a requirement, but I often use this design principle when there is any probabilistic component of my code.
  4. The researcher should know how to use their code and the libraries they use. This means I often provide very little in the way of default agents and do very little in the way of fixing the users mistakes. This often results in more work for the researcher, but I think of this as a positive.
  
  TL;DR
  1. Limited obfuscation and layer abstraction
  2. No hidden surprises/optimizations/decisions.
  3. Runtime consitency
  4. Loud errors and no free lunch.


