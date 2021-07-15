module RubyZero
    class Error < StandardError; end
    class NotImplementedError < Error; end
    class InvalidArgumentError < Error; end
    class InvalidStateError < Error; end
    class InvalidTypeError < Error; end
    class InvaildDimentionError < Error; end
    class InvaildShapeError < Error; end
end