module RubyZero::NN::Layers
    class Embedding < Affine
        def initialize(vocab_size, hidden_dim)
            super(vocab_size, hidden_dim, bias: false)
        end
    end
end