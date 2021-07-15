module RubyZero
    class TensorType
        def self.get_type(xm)
            if xm == Numo
                return get_type_numo
            elsif xm == Cumo
                return get_type_cumo
            else
                raise "#{xm} is unknown calculate library."
            end
        end
        def self.get_type_numo
            raise "#{self.class} is not defined internal Numo type."
        end

        def self.get_type_cumo
            raise "#{self.class} is not defined internal Cumo type."
        end
        
        # initialize tensor
        def self.[]*data
            Tensor.new(data, dtype:self)
        end

        # initialize random tensor
        def self.rand_norm(shape, mean=0, std=1)
            Tensor.rand_norm(shape, mean:mean, std:std, dtype:self)
        end
        # uniform random
        def self.rand_uniform(shape, low=-1, high=1)
            Tensor.rand_uniform(shape, low:low, high:high, dtype:self)
        end
    end

    # Integer types
    class Int8 < TensorType
        def self.get_type_numo
            Numo::Int8
        end
        def self.get_type_cumo
            Cumo::Int8
        end
    end

    class Int16 < TensorType
        def self.get_type_numo
            Numo::Int16
        end
        def self.get_type_cumo
            Cumo::Int16
        end
    end

    class Int32 < TensorType
        def self.get_type_numo
            Numo::Int32
        end
        def self.get_type_cumo
            Cumo::Int32
        end
    end

    class Int64 < TensorType
        def self.get_type_numo
            Numo::Int64
        end
        def self.get_type_cumo
            Cumo::Int64
        end
    end

    # Float types

    class Float32 < TensorType
        def self.get_type_numo
            Numo::Float32
        end
        def self.get_type_cumo
            Cumo::Float32
        end
    end

    class Float64 < TensorType
        def self.get_type_numo
            Numo::Float64
        end
        def self.get_type_cumo
            Cumo::Float64
        end
    end

    # Complex types
    class Complex32 < TensorType
        def self.get_type_numo
            Numo::Complex32
        end
        def self.get_type_cumo
            Cumo::Complex32
        end
    end

    class Complex64 < TensorType
        def self.get_type_numo
            Numo::Complex64
        end
        def self.get_type_cumo
            Cumo::Complex64
        end
    end

    # bit type
    class Bit < TensorType
        def self.get_type_numo
            Numo::Bit
        end
        def self.get_type_cumo
            Cumo::Bit
        end
    end

    # aliases 
    BitTensor = Bit
    ShortTensor = Int16
    IntTensor = Int32
    LongTensor = Int64
    FloatTensor = Float64
    ComplexTensor = Complex64
    
    # convert Numo dtype to RubyZero Dtype
    # xm = Numo or Cumo.
    def convert_to_rubyzero_dtype(xm, dtype)
        if xm == Numo
            return Int8 if dtype == Numo::Int8
            return Int16 if dtype == Numo::Int16
            return Int32 if dtype == Numo::Int32
            return Int64 if dtype == Numo::Int64
            return Float32 if dtype == Numo::Float32
            return Float64 if dtype == Numo::Float64
            return Complex32 if dtype == Numo::Complex32
            return Complex64 if dtype == Numo::Complex64
            return Bit if dtype == Numo::Bit
        elsif defined?(Cumo) && xm == Cumo
            return Int8 if dtype == Cumo::Int8
            return Int16 if dtype == Cumo::Int16
            return Int32 if dtype == Cumo::Int32
            return Int64 if dtype == Cumo::Int64
            return Float32 if dtype == Cumo::Float32
            return Float64 if dtype == Cumo::Float64
            return Complex32 if dtype == Cumo::Complex32
            return Complex64 if dtype == Cumo::Complex64
            return Bit if dtype == Cumo::Bit
        end
    end
end