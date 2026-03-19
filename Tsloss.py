import torch
import torch.nn as nn
from torch.nn.functional import normalize

class CustomLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(CustomLoss, self).__init__()
        self.margin = margin
        self.mse_loss = nn.MSELoss()

    def forward(self, teacher_output, student_output, labels):
        mse_loss = self.mse_loss(teacher_output, student_output)
        teacher_features = normalize(teacher_output, dim=1)
        student_features = normalize(student_output, dim=1)
        pos_mask = (labels == 1).float()
        neg_mask = 1 - pos_mask
        pos_diff = self.margin - torch.norm(teacher_features - student_features, p=2, dim=1)
        pos_loss = torch.max(pos_diff, torch.tensor(0.0).to(pos_diff.device))
        loss = torch.mean(pos_mask * pos_loss + neg_mask * mse_loss)
        return loss


if __name__ == '__main__':
    teacher_output = torch.randn(16, 512)
    student_output = torch.randn(16, 512)
    labels = torch.randint(0, 2, (16,))

    loss_fn = CustomLoss(margin=1.4)

    loss = loss_fn(teacher_output, student_output, labels)
    print('Loss:', loss.item())
