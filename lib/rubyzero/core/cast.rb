module RubyZero::Core
    # Tensor class
    class Tensor
        CAST_PRIORITY = [
            RubyZero::Core::DataTypes::Boolean,
            RubyZero::Core::DataTypes::UInt8,
            RubyZero::Core::DataTypes::Int8,
            RubyZero::Core::DataTypes::UInt16,
            RubyZero::Core::DataTypes::Int16,
            RubyZero::Core::DataTypes::UInt32,
            RubyZero::Core::DataTypes::Int32,
            RubyZero::Core::DataTypes::UInt64,
            RubyZero::Core::DataTypes::Int64,
            RubyZero::Core::DataTypes::Float32,
            RubyZero::Core::DataTypes::Float64,
            RubyZero::Core::DataTypes::Complex64,
            RubyZero::Core::DataTypes::Complex128,
        ]
        def cast_to(dtype)
            @dtype = dtype
            @data = dtype.get_type_on_calculator(@device).cast(@data)
            return self
        end
    end
end