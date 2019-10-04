

function test_construction(obj_class, args...)
    try
        obj_class(args...)
        return true
    catch y
        @show y
        return false
    end
end

