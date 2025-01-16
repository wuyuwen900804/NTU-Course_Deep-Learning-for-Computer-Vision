import time
import torch

def fit_model(model, criterion, optimizer, epochs, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, epochs, T_mult=2)
    training_loss, training_acc = [], []
    test_loss, test_acc = [], []
    best_acc = 0

    for epoch in range(epochs):
        start_time = time.time()

        # training phase
        correct_train, total_train = 0, 0
        running_train_loss = 0.0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            print(f"Batch {i+1}/{len(train_loader)} processed", end="\r")

            images = images.to(torch.float32).to(device)
            labels = labels.to(torch.int64).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()

            predicted = torch.argmax(outputs, dim=1)
            total_train += len(labels)
            correct_train += (predicted == labels).float().sum()
            running_train_loss += train_loss.item()

        train_acc = 100 * correct_train / float(total_train)
        training_acc.append(train_acc.cpu())
        training_loss.append(running_train_loss / len(train_loader))
        torch.cuda.empty_cache()

        # validation phase
        correct_test, total_test = 0, 0
        running_val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                test_images = images.to(torch.float32).to(device)
                test_labels = labels.to(torch.int64).to(device)
                outputs = model(test_images)
                val_loss = criterion(outputs, test_labels)

                predicted = torch.argmax(outputs, dim=1)
                total_test += len(test_labels)
                correct_test += (predicted == test_labels).float().sum()
                running_val_loss += val_loss.item()

        val_acc = (100 * correct_test / float(total_test)).cpu()
        test_acc.append(val_acc)
        test_loss.append(running_val_loss / len(val_loader))

        end_time = time.time()
        spend_time = end_time - start_time
        print(f'Epoch: {epoch+1}/{epochs} |',
              f'Train_loss: {training_loss[-1]:.2f}, Train_acc: {train_acc:.2f}%,',
              f'Valid_loss: {test_loss[-1]:.2f}, Valid_acc: {val_acc:.2f}%,',
              f'Spend_time: {spend_time:.2f}')
        scheduler.step()

        if val_acc > best_acc:
            torch.save(model.state_dict(), 'part1/E_best_acc.pt')
            best_acc = val_acc

    return (training_loss, training_acc, test_loss, test_acc)