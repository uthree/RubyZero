module RubyZero
    class Tensor
        attr_reader :xm, :dtype
        attr_accessor :data, :grad, :require_grad, :grad_fn, :generation, :trainable
        def initialize(data, require_grad:false, grad:nil, generation:0, xm:Numo, dtype:nil, traineble:false)
            if data.is_a? xm::NArray # NArray系列の場合はそのままデータに代入
                @data = data
            elsif data.is_a?(Array) # 配列の場合はNumo::NArrayに変換
                if dtype.nil?
                    @data = xm::NArray[*data]
                else
                    @data = dtype.get_type(xm)[*data]
                end
            elsif data.is_a?(Tensor) # Tensorの場合はコピーしてデータに代入
                @data = data.data.dup
            elsif data.is_a?(Integer) # 整数の場合はNumo::NArrayに変換
                @data = xm::NArray[data]
            elsif data.is_a?(Float) # 浮動小数点数の場合はNumo::NArrayに変換
                @data = xm::NArray[data]
            elsif data.is_a?(Complex) # 複素数の場合はNumo::NArrayに変換
                @data = xm::NArray[data]
            else # 変換できないのでエラーにする
                raise TypeError, "unsupported type #{data.class} for #{self.class}"
            end
            
            @xm = xm
            update_dtype()
            @grad_fn = nil
            @grad = nil
            @generation = generation
            @require_grad = require_grad
            @trainable = trainable
        end

        def id
            return object_id
        end

        def trainable?
            return @trainable
        end

        # 勾配を追加する
        def add_grad(other_tensor, require_grad:false)
            @grad ||= zeros_like
            @grad.require_grad = require_grad
            @grad.data += other_tensor.data
        end

        def init_gradients()
            @grad = zeros_like
        end

        # dtypeを更新する。
        def update_dtype()
            @dtype = convert_to_rubyzero_dtype(@xm, @data.class)
        end

        # initialize with bracket
        def self.[](*data)
            Tensor.new(data)
        end

        # initialize zeros
        def self.zeros(shape, dtype)
            internal_type = dtype.get_type(Numo)
            data = internal_type.zeros(shape)
            return new(data, dtype:dtype)
        end

        # initialize ones
        def self.ones(shape, dtype)
            internal_type = dtype.get_type(Numo)
            data = internal_type.ones(shape)
            return new(data, dtype:dtype)
        end

        # initialize zeros with same shape
        def zeros_like
            data = self.dtype.get_type(self.xm).zeros(*self.shape)
            Tensor.new(data, require_grad:@require_grad, grad:@grad, generation:@generation, xm:@xm, dtype:@dtype)
        end

        # initialize zeros with same shape
        def ones_like
            data = self.dtype.get_type(self.xm).ones(*self.shape)
            Tensor.new(data, require_grad:@require_grad, grad:@grad, generation:@generation, xm:@xm, dtype:@dtype)
        end
        
        def shape
            return @data.shape
        end

        def ndim
            if @data.ndim == 1 and @data.shape[0] == 1
                return 0
            else
                return @data.ndim
            end
        end

        def inspect
            return "#< RubyZero::Tensor dtype=#{dtype} shape=#{shape} grad_fn=#{@grad_fn.class} \n (#{@data.inspect})>"
        end

        def item
            if ndim == 0
                return @data[0]
            else
                return @data.to_a
            end
        end
    end
end