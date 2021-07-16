module RubyZero
    class Tensor
        def repeat_to(other)
            a, b = self, other
            if a.shape.size > b.shape.size
                a, b = b, a
            end

            return a if shape == b.shape
            if ndim > 0
                raise InvaildShapeError, "Inner shape is mismatch." unless shape == other.shape[ndim..-1]
            end
            repeat_shape = other.shape[0..ndim-1]
            r = self
            repeat_shape.reverse.each do |s|
                r = self.repeat(s)
            end
            return r
        end

        def repeat_to!(other)
            self.data = repeat_to(other).data
        end
    end
end