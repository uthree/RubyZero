module RubyZero
    class Error < StandardError; end
    class NotImplementedError < Error; end
    class InvalidArgumentError < Error; end
    class InvalidStateError < Error; end
    class InvalidTypeError < Error; end
    class InvalidValueError < Error; end
end