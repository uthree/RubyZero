module RubyZero::Core
    class Tensor
        def backward()
            return unless @grad_fn

            @grad ||= ones_like
            dx = @grad_fn.backward(@grad)
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