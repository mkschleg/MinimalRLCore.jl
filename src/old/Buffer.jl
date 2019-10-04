# module Buffer

# This is really similar to the CircularBuffer implementation, but has _some_ nice things that aren't available.
# Ideally we might want to reimplement _data_frame with a DataStructures Circular Buffer to make this implementation simpler.

# Although, it might be better to keep this delinked as the data_frame implementation allows for inline memory of the arrays.

using DataFrames
import Base.size, Base.getindex

export CircularBuffer, add!, size, capacity, getindex

mutable struct CircularBuffer
    _data_frame::DataFrame
    _current_col::Int64
    _capacity::Int64
    _full::bool
    _data_types::Array
    function CircularBuffer(size, types; column_names=nothing)
        if column_names != nothing
            data_frame = DataFrame(types, column_names, size)
        else
            data_frame = DataFrame(types, size)
        end
        new(data_frame, 1, size, false, types)
    end
    function CircularBuffer(args...; kwargs...)
        data_frame = DataFrame(args...; kwargs...)
        new(data_frame, 1, size(data_frame)[1], false, eltypes(d))
    end
end

# function _set_row

function add!(buffer::CircularBuffer, experience)
    ret = buffer._current_col
    for (idx, dat) in enumerate(experience)
        buffer._data_frame[buffer._current_col, idx] = dat
    end
    buffer._current_col += 1
    if buffer._current_col > buffer._capacity
        buffer._current_col = 1
        buffer._full = true
    end
    return ret
end

function size(buffer::CircularBuffer)
    if (buffer._full)
        size(buffer._data_frame)
    else
        (buffer._current_col-1, length(buffer._data_types))
    end
end
capacity(buffer::CircularBuffer) = buffer._capacity
getindex(buffer::CircularBuffer, idx) = getindex(buffer._data_frame, idx)
getrow(buffer::CircularBuffer, idx) = buffer._data_frame[idx,:]
# get(idx::Array{Int64, 1}) = d[idx]

# end
