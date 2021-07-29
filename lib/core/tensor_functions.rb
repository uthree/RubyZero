require_relative "./tensor.rb"
require_relative "../functions/tensor_function.rb"

module RubyZero
    class Tensor
        def reshape(*shape)
            return Functions::ReShape.new(*shape).call(self)
        end

        def transpose(*axes)
            return Functions::Transpose.new(*axes).call(self)
        end

        def swap_axes(*axes)
            return Functions::SwapAxes.new(*axes).call(self)
        end
        alias_method :swapaxes, :swap_axes

        def repeat(repeats, axis:0)
            # どの軸にもRepeatできるように拡張したもの
            shape=self.shape.dup.insert(axis, repeats)
            target_shape = self.shape.dup.insert(axis, repeats)
            return Functions::RepeatZeroAxis.new(repeats).call(self.swap_axes(axis, 0)).reshape(*target_shape).swap_axes(0, axis)
        end

        # 総和
        def sum(axis:0)
            return Functions::Sum.new(axis).call(self)
        end

        def mean(axis:0)
            return Functions::Mean.new(axis).call(self)
        end

        #行列積
        def dot(other)
            raise InvaildShapeError, "shape mismatch" if shape[1] != other.shape[0]
            raise InvaildDimentionError, "both tensor of #{self.class}.dot(other) must be ndim=2" if ndim != 2 or other.ndim != 2
            return Functions::MatMul.new.call(self, other)
        end

        #slice
        def [](*args)
            return Functions::Slice.new(*args).call(self)
        end

        # assingment with slice
        def []=(*args,val)
            result = Functions::Assign.new(*args).call(self, val)
            # copy to self
            @data = result.data
            @grad = result.grad
            @require_grad = result.require_grad
            @grad_fn = result.grad_fn
            @generation = result.generation
            return result
        end


        #flatten 
        def flatten(start_axis:0, end_axis:1)
            new_length = self.shape[start_axis..end_axis].reduce(&:*)
            new_shape = self.shape.dup
            new_shape.slice!(start_axis..end_axis)
            new_shape.insert(start_axis, new_length)
            return self.reshape(*new_shape)
        end

        #concat
        def cat(other, axis:0)
            return Functions::Concatenate.new(axis:axis).call(self, other)
        end

        # aliases of concatenate
        alias_method :concat, :cat
        alias_method :concatenate, :cat

        #stack
        def self.stack(tensors, axis:0) # TODO: optimze to O(log2(n))
            first_tensor = tensors.first
            tensors[1..-1].each do |tensor|
                first_tensor = first_tensor.cat(tensor, axis:axis)
            end
            return first_tensor
        end
    end
end