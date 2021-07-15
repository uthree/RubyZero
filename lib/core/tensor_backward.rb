module RubyZero
    class Tensor
        def backward(require_grad: false) # TODO: 再帰所為なので、後でループにする。
            @grad ||= ones_like()
            if @grad_fn
                inputs = @grad_fn.inputs
                grads = @grad_fn.backward(@grad)
                grads.length.times do |i|
                    grads[i].require_grad = require_grad
                    inputs[i].add_grad(grads[i], require_grad: require_grad)
                    inputs[i].backward(require_grad: require_grad)
                end
            end
        end
    end
end