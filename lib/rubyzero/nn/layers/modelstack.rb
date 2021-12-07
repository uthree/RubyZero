module RubyZero::NN::Layers
    # The stack of layers
    # This class don't supports to containing different Model classes.
    class ModelStack < ModelList
        def <<(layer)
            if @models.size > 0
                if layer.class != @models.last.class
                    raise TypeError, "Layer type mismatch"
                end
            end
            super(layer)
        end
        def forward(*args)
            @models.each do |model|
                args = model.forward(*args)
            end
            return args
        end
    end
end