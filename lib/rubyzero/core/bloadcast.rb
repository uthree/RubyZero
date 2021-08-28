module RubyZero::Core
    class Tensor
        # Broadcast a tensor to a new shape
        def bloadcast_to(other)
            if other.kind_of? Tensor
                repeat_axes = other.shape.axes[self.ndim..-1]
                repeat_axes.each do |axis|
                    self.repeat(axis.size, axis: axis.index)
                end
            elsif other.kind_of? Numeric

            end
        end
    end
end