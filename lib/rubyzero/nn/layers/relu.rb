module RubyZero::NN::Layers
    class ReLU < RubyZero::NN::Model
        def forward(x)
            F.relu(x)
        end
    end
end