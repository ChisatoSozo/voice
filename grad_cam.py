import torch


def guided_backprop(model, images, target_class):
    model.eval()
    output = model(images)

    # zero gradients
    model.zero_grad()

    # Target for backprop
    one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
    one_hot_output[0][target_class] = 1

    # Backward pass
    output.backward(gradient=one_hot_output)

    return images.grad


def register_relu_hooks(model):
    """
    Function to register hooks onto ReLU modules to modify their gradients during the backward pass.
    """
    def relu_backward_hook_function(module, grad_in, grad_out):
        """
        If there is a negative gradient, change it to zero
        """
        return (torch.clamp(grad_in[0], min=0.0),)

    # Loop through the modules in the model
    for pos, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            module.register_full_backward_hook(relu_backward_hook_function)


def make_guided_backprop(model):
    # Set the gradients to zero
    for param in model.parameters():
        param.requires_grad = False

    # Apply guided backpropagation
    register_relu_hooks(model)

    def do_guided_backprop(image, target_class):
        images = image.unsqueeze(0)
        images.requires_grad = True
        activation_map = guided_backprop(
            model, images, target_class)
        return activation_map

    return do_guided_backprop
