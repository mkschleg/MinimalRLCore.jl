# MinimalRLCore.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://mkschleg.github.io/MinimalRLCore.jl/dev)
[![Build Status](https://travis-ci.com/mkschleg/JuliaRL.jl.svg?branch=refactor)](https://travis-ci.com/mkschleg/MinimalRLCore.jl)


This is the core to many of my RL frameworks in Julia. It follows some wisdom brought to us by Adam White and Brian Tanner in their development of RLGlue. Some key attributes:

- Hard seperation of Agent, Environment, and Experiment concepts as separate type trees.
- Minimal api and few opinionated decisions with more complete experiments and ideas in separate sister repositories.
- Minimal implementation to allow for as much mucking around as possible.


## Prior Work

The [ReinforcementLearningBase.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl) package is another really nice project which works to implement core reinforcement learning concepts for Julia. This is a nice set of interfaces for core RL topics. The core design principle of the two packages seems different and I believe the overall goals of the projects are quite different. Another package apart of the JuliaML group is [Reinforce.jl](https://github.com/JuliaML/Reinforce.jl). Issues like not passing the agent when the environment is terminal makes this unusable for my (and others in my group) research. RLCore also only focuses on defining the environment-agent interaction and tries not have any opinions on how learning happens or the types of information the agent needs to decide on an action.

### Core differences

- Allow the user to tinker as much as possible, and allow "non-default ideas": One example of this is found in the agent and episode interface. While we can reliably guess what the environment will return (and the environment should actually return only the typical state, reward, and termination), the agent can return all sorts of information that the user might want to work with. Taking this idea, the framework expects the agent to return either the action to take next or a `NamedTuple` w/ at least one `action` component. The action will be passed to the environment, and the full NamedTuple will be passed to the user in the experience tuple, which the user can do w/ as they wish. This is an important concept for prediction experiments where often we want to return the prediction the agent makes w/ the next action.
- Allow the user to manage there own RNG. This is important when running many experiments in threads (pre v1.3) to make sure the results will be consistent. This also allows users to pass in specific starting states for an environment to start w/ for monte-carlo rollouts.
- Inclusion of GVF specification functions.


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
  4. Loud errors


