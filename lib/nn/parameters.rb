module RubyZero::NN
    class Parameters
        attr_reader :elements
        def initialize(*elements)
            @elements = elements
            train
            @mode = :train
        end
        def train
            @elements.each do |element|
                element.require_grad = true
            end
            @mode = :train
        end
        def eval
            @elements.each do |element|
                element.require_grad = false
            end
            @mode = :eval
        end
        def eval?
            return @mode == :eval
        end
        def train?
            return @mode == :train
        end
        def << (element)
            @elements << element
            return nil
        end
        def + (other)
            @elements + other.elements
            return self
        end
    end
end