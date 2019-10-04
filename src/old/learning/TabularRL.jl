module TabularRL

using Lazy
import Base.getindex, Base.setindex!

import ..AbstractVFunction
import ..AbstractQFunction
import ..update!

export VFunction, QFunction

"""
    VFunction(dims::Integer...; init)
Tabular V Function
"""
mutable struct VFunction{W} <: AbstractVFunction
    values::W
end
VFunction(dims::Integer...; init=zeros) = VFunction(init(dims...))

weights(v::VFunction) = v.values
update!(value::VFunction, s, δ) = value[s] .+= δ

@forward VFunction.values Base.getindex
@forward VFunction.values Base.setindex!

#---------------------#
#
# State-Action Value Functions
#
#---------------------#

"""
    QFunction
Tabular Q Function
"""
mutable struct QFunction{W} <: AbstractQFunction
    values::W
end

"""
    QFunction(num_actions, state_dims...)
Creates a action-state value function of size num_actions x state_dims...
"""
QFunction(num_actions::Integer, state_dims::Integer...; init=zeros) = QFunction(init(num_actions, state_dims...))

@forward QFunction.values getindex
@forward QFunction.values setindex!


end # module TabularRL
