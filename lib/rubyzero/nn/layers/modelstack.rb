module RubyZero::NN::Layers
    # the stack of layers
    class ModelStack < ModelList
        def <<(layer)
            if @models.size > 0
                if layer.class != @models.last.class
                    raise TypeError, "Layer type mismatch"
                end
            end
            super(layer)
        end
    end
end