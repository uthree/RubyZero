module RubyZero::Utils
    class Trainer
        def iniitialize(loss_fn, optimizer, device)
            @loss_fn = loss_fn
            @optimizer = optimizer
            @device = device
        end
        def train(model, train_loader, epochs)
            
        end
    end
end