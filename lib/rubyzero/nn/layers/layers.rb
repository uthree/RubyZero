module RubyZero::NN
    module Layers
        
    end
end


require_relative './layer.rb'
require_relative './affine.rb'
require_relative './relu.rb'
require_relative './modellist.rb'
require_relative './modelstack.rb'
require_relative './multi_layer_perceptron.rb'
require_relative './embedding.rb'

module RubyZero::NN
    module LayersInitializeAliases
        def self.affine(*args)
            RubyZero::NN::Layers::Affine.new(*args)
        end
        def self.relu(*args)
            RubyZero::NN::Layers::ReLU.new(*args)
        end
        def self.model_list(*args)
            RubyZero::NN::Layers::ModelList.new(*args)
        end
        def self.model_stack(*args)
            RubyZero::NN::Layers::ModelStack.new(*args)
        end
        def self.multi_layer_perceptron(*args)
            RubyZero::NN::Layers::MultiLayerPerceptron.new(*args)
        end
        def self.mlp(*args)
            RubyZero::NN::Layers::MultiLayerPerceptron.new(*args)
        end
        def embedding(*args)
            RubyZero::NN::Layers::Embedding.new(*args)
        end
    end
end

module RubyZero
    L = RubyZero::NN::LayersInitializeAliases
end
