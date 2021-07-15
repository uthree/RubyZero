module RubyZero::Functions
    class ReLU < Function
        def forward(x)
            @pass_grad_mask = x.data > 0
            x_type = x.dtype.get_type(x.xm)
            @pass_grad_mask = @pass_grad_mask.cast_to(x_type)
            output = x.data * @pass_grad_mask
            return Tensor.new(output)
        end
        def backward(grad)
            data = @pass_grad_mask * grad.data
            return [ Tensor.new(data) ]
        end
    end
end