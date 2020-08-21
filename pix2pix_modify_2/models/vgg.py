import torch.nn as nn
import torchvision.models as models
import torch    

class VGG(nn.Module):
    '''
    Classic pre-trained VGG19 model.
    Its forward call returns a list of the activations from
    the predefined content layers.
    '''

    def __init__(self, gpu_ids, layer_names, content_layers):
        super(VGG, self).__init__()
        if len(gpu_ids) > 0:
            torch.cuda.set_device(gpu_ids[0])
        self.layer_names = layer_names
        self.content_layers = content_layers

        vgg_19 = models.vgg19(pretrained=True)
        for param in vgg_19.parameters():
            param.requires_grad = False
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            vgg_19.to(gpu_ids[0])
            # vgg_19 = torch.nn.DataParallel(vgg_19, gpu_ids)

        features = vgg_19.features

        self.features = nn.Sequential()
        for i, module in enumerate(features):
            name = self.layer_names[i]
            self.features.add_module(name, module)



    def forward(self, input):
        batch_size = input.size(0)
        all_outputs = []
        output = input
        for name, module in self.features.named_children():
            output = module(output)
            if name in self.content_layers:
                all_outputs.append(output.view(batch_size, -1))
        return all_outputs