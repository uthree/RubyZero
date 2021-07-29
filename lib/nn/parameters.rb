module RubyZero::NN
    class Parameters
        attr_reader :elements, :mode
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
        def +(other)
            Parameters.new(*(@elements + other.elements))
        end
        def [](index)
            return @elements[index]
        end
        def []=(index, value)
            return @elements[index] = value
        end
        def each
            @elements.each do |element|
                yield element
            end
        end
        include Enumerable
    end
end