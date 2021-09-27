module RubyZero::Core
    class Tensor
        def backward()
            @grad ||= ones_like
            return unless @grad_fn
            dx = @grad_fn.backward(self)
            self.grad_fn.inputs.each_with_index do |input, i|
                if input.grad 
                    input.grad += dx[i]
                else
                    input.grad = dx[i]
                end
                input.backward()
            end
        end
    end
end