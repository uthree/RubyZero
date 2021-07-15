module RubyZero::NN
    class Parameters
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
        end
        def + (other)
            @elements += other.elements
        end
    end
end