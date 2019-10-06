# RLCore.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://mkschleg.github.io/RLCore.jl/dev)
[![Build Status](https://travis-ci.com/mkschleg/JuliaRL.jl.svg?branch=refactor)](https://travis-ci.com/mkschleg/RLCore.jl)


This is the core to many of my RL frameworks in Julia. It follows some wisdom brought to us by Adam White and Brian Tanner in their development of RLGlue. Some key attributes:

- Options to manage random number generator to ensure reproducibility.
- Hard seperation of Agent, Environment, and Experiment concepts as seperate type trees.
- Minimal api with more complete experiments and ideas in sperate sister repositories.

