import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# from dataset import PascalVOC2012Dataset, collate_fn
# from model import Yolov1
# from loss import YOLOv1Loss
# from train import train_model, validate_model, load_checkpoint, save_checkpoint

seed = 123
torch.manual_seed(seed)

# Hyperparameter
C = 20
B = 2
S = 7
LEARNING_RATE  = 1e-4
BATCH_SIZE     = 4
EPOCHS         = 1
NUM_WORKERS    = 2
PIN_MEMORY     = True
IMG_SIZE       = 448
WEIGHT_DECAY   = 5e-4
MOMENTUM       = 0.9
LOAD_MODEL     = True
ROOT_DIR       = "D:\\MACHINE LEARNING\\MODELS\\YOLOv1\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    # initialize train_dataset
    train_dataset = PascalVOC2012Dataset(
        root_dir = ROOT_DIR,
        split    = "train",
        S        = S,
        B        = B,
        C        = C
    )

    # initialize train_loader
    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = NUM_WORKERS,
        pin_memory  = PIN_MEMORY,
        collate_fn  = collate_fn,
    )

    # initialize eval_dataset
    valid_dataset = PascalVOC2012Dataset(
        root_dir = ROOT_DIR,
        split    = "val",
        S        = S,
        B        = B,
        C        = C
    )

    # initialize train_loader
    val_loader = DataLoader(
        valid_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = PIN_MEMORY,
        collate_fn  = collate_fn,
    )

    # initialize model, loss, opimizer
    model     = Yolov1(split_size=S, num_boxes=B, num_classes=C).to(DEVICE)
    criterion = YOLOv1Loss(S=S,B=B, C=C).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # Load model from checkpoint if LOAD_MODEL is True
    start_epoch = 0

    # Training and validation loop
    best_loss = float("inf")
    for epoch in range(start_epoch, EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        train_loss = train_model(model, train_loader, criterion, optimizer, DEVICE, epoch)
        val_loss = validate_model(model, val_loader, criterion, DEVICE)
        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

        # Save checkpoint if validation loss improves
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, filepath="best_model.pth")

if __name__ == '__main__':
    main()


# PLOT IMAGE
for idx in range(train_dataset.__len__()):
    original_img, img, target = train_dataset.__getitem__(idx)
    output = model(img)
    bboxes = []
    labels = []
    confs  = []
    factor = 1. / S
    W, H = original_img.size

    for i in range(S):
        for j in range(S):
            for k in range(B):
                #print(output.shape)
                out  = output[0, i, j, k*5:k*5+5]
                #print(out.shape)
                #print(output)
                box  = out[:4]
                conf = out[4]
                cls  = output[0, i, j, B*5:]
                #print(cls.shape)
                a, b = torch.max(cls, dim=-1)
                conf = a*conf
                label = b
                #print(label)
                #sys.exit()
                if (conf < 0.3):
                    continue
                cx_cell = box[0]
                cy_cell = box[1]
                w       = box[2]
                h       = box[3]

                cx = (cx_cell + j) * factor
                cy = (cy_cell + i) * factor

                #x1 = int((cx - w/2) * W)
                #y1 = int((cy - h/2) * H)
                #x2 = int((cx + w/2) * W)
                #y2 = int((cy + h/2) * H)
                x1 = (cx - w/2)
                y1 = (cy - h/2)
                x2 = (cx + w/2)
                y2 = (cy + h/2)

                bboxes.append([x1, y1, x2, y2])
                confs.append(conf)
                labels.append(label)

    if len(bboxes) > 0:
        bboxes = torch.tensor(bboxes)
        confs  = torch.tensor(confs)
        labels = torch.tensor(labels)
        mask = torchvision.ops.nms(bboxes, confs, 0.5)

        bboxes = bboxes[mask]
        labels = labels[mask]
        draw = ImageDraw.Draw(original_img)
        font = ImageFont.load_default()

        for (box, label) in zip(bboxes, labels):
            x1 = int(box[0] * W)
            y1 = int(box[1] * H)
            x2 = int(box[2] * W)
            y2 = int(box[3] * H)
            #x1 = box[0]
            #y1 = box[1]
            #x2 = box[2]
            #y2 = box[3]
            #print(x1, y1, x2, y2)

            draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=2)
            draw.text((x1, y1), idx2name[int(label.item())], font=font)
    
        plt.imshow(original_img)
        plt.show()