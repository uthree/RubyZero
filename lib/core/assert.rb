require_relative "../core/tensor.rb"

module RubyZero
    class Tensor
        def assert_shape(expected_shape)
            expected_shape.each_with_index do |expected_axis, i|
                if expected_axis[i] != nil
                    if shape[i] != expected_axis
                        raise InvaildShapeError, "Expected shape #{expected_shape}, got #{shape}"
                    end
                end
            end
            return true
        end
        def assert_ndim(expected_dim)
            if ndim == expected_dim
                return true
            else
                raise InvaildShapeError, "Expected ndim #{expected_dim}, got #{ndim}"
            end
        end
    end
end