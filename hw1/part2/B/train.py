import time
import torch
import numpy as np
from mean_iou_evaluate import mean_iou_score

def fit_model(model, criterion, optimizer, epochs, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader), eta_min=1e-6)
    training_loss, training_iou = [], []
    test_loss, test_iou = [], []
    best_iou = 0

    for epoch in range(epochs):
        start_time = time.time()

        # training phase
        train_prediction_list, train_masks_list = [], []
        train_iou, val_iou = 0, 0
        model.train()
        for i, (images, masks) in enumerate(train_loader):
            print(f"Batch {i+1}/{len(train_loader)} processed", end="\r")

            images = images.to(torch.float32).to(device)
            masks = masks.to(torch.float32).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            train_loss = criterion(outputs, masks.long())
            train_loss.backward()
            optimizer.step()

            prediction = torch.max(outputs, 1)[1]
            train_prediction_list.extend(prediction.cpu().numpy())
            train_masks_list.extend(masks.cpu().numpy())
        
        print()
        train_iou = mean_iou_score(np.array(train_prediction_list), np.array(train_masks_list))
        training_loss.append(train_loss.data.cpu())
        training_iou.append(train_iou)
        torch.cuda.empty_cache()

        # validation phase
        val_prediction_list, val_masks_list = [], []
        model.eval()
        with torch.no_grad():
            for i, (images, masks) in enumerate(val_loader):
                test_images = images.to(torch.float32).to(device)
                test_masks = masks.to(torch.int64).to(device)

                outputs = model(test_images)
                val_loss = criterion(outputs, test_masks)

                prediction = torch.max(outputs, 1)[1]
                val_prediction_list.extend(prediction.cpu().numpy().flatten())
                val_masks_list.extend(masks.cpu().numpy().flatten())

        val_iou = mean_iou_score(np.array(val_prediction_list), np.array(val_masks_list))
        test_loss.append(val_loss.data.cpu())
        test_iou.append(val_iou)

        end_time = time.time()
        spend_time = end_time - start_time

        print(f'Epoch: {epoch+1}/{epochs} |',
              f'Train_loss: {train_loss.data:.2f}, Train_iou: {train_iou:.2f},',
              f'Valid_loss: {val_loss.data:.2f}, Valid_iou: {val_iou:.2f},',
              f'Spend_time: {spend_time:.2f}')
        scheduler.step()

        if val_iou > best_iou:
            torch.save(model.state_dict(), 'part2/B_best_iou.pt')
            best_iou = val_iou

        if (epoch+1) == 1 or (epoch+1) == (epochs/2) or (epoch+1) == epochs:
            torch.save(model.state_dict(), f'part2/B_EPOCH_{epoch+1}.pt')

    return (training_loss, training_iou, test_loss, test_iou)