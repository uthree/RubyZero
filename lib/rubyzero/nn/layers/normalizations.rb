module RubyZero::NN::Layers
    class BatchNormalization < RubyZero::NN::Layers::Layer
        def initialize(epsilon: 1e-5, mu: 0.0, sigma: 1.0)
            @epsilon = RubyZero::Float32[epsilon]
            @mu = RubyZero::Float32[mu]
            @sigma = RubyZero::Float32[sigma]
        end

        def forward(x)
            m = x.mean(axis:0)
            if x.requires_grad?
                s = F.sqrt((x * x).mean(axis:0))
                @sigma = s
            else
                s = @sigma
            end
            x_hat = (x - m) / (s + @epsilon)
            x_hat += @mu
            return x_hat
        end
    end
end