module RubyZero::NN::Layers
    # The list of models that can be used in the neural network.
    class ModelList < Layer
        def initialize(list=nil)
            if list
                @models = list
            else
                @models = []
            end
        end
        def parameters
            params = []
            @models.each do |model|
                params += model.parameters.elements
            end
            return RubyZero::NN::Parameters.new(params)
        end
        def <<(model)
            @models << model
        end
        def elements
            return @models
        end
        def +(modellist)
            return ModelList.new(@models + modellist.elements)
        end
        def each(&block)
            @models.each(&block)
        end
    end
end