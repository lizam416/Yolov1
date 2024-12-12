import torch
import torch.nn as nn

def local_to_global(box, grid, factor):
    # box  [T, b, 4] [x_cell, y_cell, w, h] -> [x1, y1, x2, y2]
    # grid [T, 2] [x, y]

    grid = grid.unsqueeze(1)

    x = box[..., [0]] * factor + grid[..., [0]]
    y = box[..., [1]] * factor + grid[..., [1]]
    w = box[..., [2]]
    h = box[..., [3]]

    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2

    return torch.cat([x1, y1, x2, y2], dim=-1)


class YOLOv1Loss(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super(YOLOv1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.eps = 1e-8
        self.mse = nn.MSELoss(reduction='sum')

    def compute_iou(self, pred_bboxes, truth_bboxes, grid):
        factor       = 1. / self.S
        # [T, bb, 4]
        pred_bboxes  = local_to_global(pred_bboxes, grid, factor)
        truth_bboxes = local_to_global(truth_bboxes, grid, factor)

        # [T, 1, 1, 4]
        truth_bboxes = truth_bboxes[:, [0], :].unsqueeze(2)
        na           = truth_bboxes.shape[1]
        # [T, 1, 2, 4]
        pred_bboxes  = pred_bboxes.unsqueeze(1)
        nb           = pred_bboxes.shape[2]

        truth_bboxes = truth_bboxes.repeat([1, 1, nb, 1])

        x1_truth = truth_bboxes[..., [0]]
        y1_truth = truth_bboxes[..., [1]]
        x2_truth = truth_bboxes[..., [2]]
        y2_truth = truth_bboxes[..., [3]]

        x1_pred = pred_bboxes[..., [0]]
        y1_pred = pred_bboxes[..., [1]]
        x2_pred = pred_bboxes[..., [2]]
        y2_pred = pred_bboxes[..., [3]]


        #print(x1_truth.shape)
        #print(x1_pred.shape)

        x1_max = torch.max(x1_truth, x1_pred)
        y1_max = torch.max(y1_truth, y1_pred)
        x2_min = torch.min(x2_truth, x2_pred)
        y2_min = torch.min(y2_truth, y2_pred)

        w = (x2_min - x1_max).clamp(0)
        h = (y2_min - y1_max).clamp(0)

        intersect = w*h

        area_a = (y2_truth - y1_truth) * (x2_truth - x1_truth)
        area_b = (y2_pred - y1_pred) * (x2_pred - x1_pred)

        union = area_a + area_b - intersect

        return intersect / union

    ###########

    def __call__(self, predict,  truth):
        # predict [Batch, S, S, (20 + B*5)] [8, 7, 7, 30]
        # truth    same

        batch = predict.shape[0]

        factor = 1. / self.S
        coor   = torch.arange(0, 1., factor).to('cuda')
        coor_y, coor_x   = torch.meshgrid([coor, coor], indexing='ij')

        grid = torch.stack([coor_x, coor_y], dim=-1).view(-1, 2)
        grid = grid.unsqueeze(0).unsqueeze(0).repeat([predict.shape[0], 1, 1, 1])
        grid = grid.view(-1, 2)

        # C + B*5
        N = predict.shape[-1]

        # [T, N]
        predict = predict.view(-1, N)
        truth   = truth.view(-1, N)

        # [T, 1]
        obj_mask   = (truth[..., 4] == 1)
        noobj_mask = (truth[..., 4] == 0)

        pred_bboxes  = []
        truth_bboxes = []
        pred_conf    = []
        for b in range(self.B):
            pred_bboxes.append(predict[..., b*5:b*5+4])
            truth_bboxes.append(truth[..., b*5:b*5+4])
            pred_conf.append(predict[..., b*5+4])

        # [T, 2, 4]
        pred_bboxes  = torch.stack(pred_bboxes, dim=1) # [Batch * S * S, B, 4]
        pred_conf    = torch.stack(pred_conf, dim=1) # [Batch * S * S, B]
        truth_bboxes = torch.stack(truth_bboxes, dim=1) # [Batch * S * S, B, 4]
        #print(truth_bboxes[obj_mask])
        #sys.exit()

        with torch.no_grad():
            iou = self.compute_iou(pred_bboxes, truth_bboxes, grid) # [Batch * S * S, B]
            iou = iou.squeeze(-1) #

        # [392, 1], [392, 1]
        max_val_iou, max_idx_iou = torch.max(iou, dim=-1)
        max_val_iou.squeeze_(-1)
        max_idx_iou.squeeze_(-1)
        #print(pred_bboxes.shape, pred_conf.shape)
        #print(max_val_iou.shape, max_idx_iou.shape)

        tt = torch.arange(pred_bboxes.shape[0])

        pred_bboxes = pred_bboxes[tt, max_idx_iou]
        pred_conf   = pred_conf[tt, max_idx_iou]
        pred_cls    = predict[..., 5*self.B:]
        #print(pred_bboxes.shape)
        #sys.exit()

        truth_bboxes = truth_bboxes[tt, max_idx_iou]
        truth_cls    = truth[..., self.B * 5:]

        term1 = self.mse(pred_bboxes[obj_mask][:, 0], truth_bboxes[obj_mask][:, 0]) + self.mse(pred_bboxes[obj_mask][:, 1], truth_bboxes[obj_mask][:, 1])

        term2 = (
            self.mse(torch.sqrt(pred_bboxes[obj_mask][:, 2] + self.eps), torch.sqrt(truth_bboxes[obj_mask][:, 2] + self.eps)) +
            self.mse(torch.sqrt(pred_bboxes[obj_mask][:, 3] + self.eps), torch.sqrt(truth_bboxes[obj_mask][:, 3] + self.eps))
        )

        term3 = self.mse(pred_conf[obj_mask], max_val_iou[obj_mask])

        term4 = self.mse(pred_conf[noobj_mask], torch.zeros_like(pred_conf[noobj_mask]))

        term5 = self.mse(pred_cls[obj_mask], truth_cls[obj_mask])

        #print(pred_bboxes[obj_mask][:, 0])
        #print(pred_bboxes[obj_mask][:. 2])
        #print('#################')
        #print(truth_bboxes[obj_mask][:, 0])
        #print(truth_bboxes[obj_mask][:, 2])

        loss = self.lambda_coord * (term1 + term2) + term3 + self.lambda_noobj * term4 + term5
        loss /= batch
        #print(loss.item())
        #print(obj_mask.sum())
        return loss