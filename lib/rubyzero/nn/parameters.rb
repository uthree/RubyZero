module RubyZero::NN
    class Parameters
        attr_accessor :elements
        def initialize(elems)
            @elements = elems
        end
        def each(&block)
            @elements.each(&block)
        end
        def <<(element)
            @elements << element
        end
        def size
            sz = 0
            @elements.each do |e|
                sz += e.shape.inject(:*)
            end
            return sz
        end
    end
end