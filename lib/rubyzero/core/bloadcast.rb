module RubyZero::Core
    class Tensor
        # Broadcast a tensor to a new shape
        def bloadcast_to(other)
            if other.kind_of? Tensor
                raise ShapeMissmatchError, "Cannot broadcast shape=#{self.shape.to_a} to shape=#{self.shape.to_a}" unless other.shape.axes[self.ndim..-1] == other.shape.to_a
                repeat_axes = other.shape.axes[self.ndim..-1]
                repeat_axes.each do |axis|
                    self.repeat(axis.size, axis: axis.index)
                end
            end
        end

        # Broadcast two tensors to same shape
        # @param x1 [Tensor]
        # @param x2 [Tensor]
        def self.bloadcast_to_same(x1, x2)
            x1 = Tensor[x1] unless x1.kind_of? Tensor
            x2 = Tensor[x2] unless x2.kind_of? Tensor
            if x1.ndim < x2.ndim
                x1.bloadcast_to(x2)
            end
            if x2.ndim < x1.ndim
                x2.bloadcast_to(x1)
            end
            return x1, x2
        end
    end
end