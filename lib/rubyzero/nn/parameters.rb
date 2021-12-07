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
        def to_marshal()
            return Marshal.dump(@elements.map{|e| e.data})
        end
        def self.from_marshal(marshal)
            return Parameters.new(Marshal.load(marshal).map{|d| RubyZero::Core::Tensor.new(d)})
        end
        def save(path)
            File.open(path, 'wb') do |f|
                f.write(to_marshal)
            end
        end
        def self.load(path)
            return from_marshal(File.read(path))
        end
        include Enumerable
    end
end