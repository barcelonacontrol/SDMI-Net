import torch
from torch.nn.functional import kl_div

MySoftmax = torch.nn.Softmax(dim=1)

def feat_label_mi_estimation(model, Feat, Y):
    temperature = 0.05
    # pred_Y = model.out_conv(Feat)
    mi = kl_div(input=MySoftmax(Feat.detach() / temperature), target=MySoftmax(Y / temperature),
                reduction='mean')  # pixel-level

    return mi


def feat_feat_mi_estimation(F1, F2):
    """
        F1 -> F2
    """
    temperature = 0.05
    mi = kl_div(input=MySoftmax(F1.detach() / temperature), target=MySoftmax(F2 / temperature))
    return mi