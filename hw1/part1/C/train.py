import time
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
set_seed(0)

def get_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, label in dataloader:
            images = images.to(torch.float32).to(device)
            output = model.avgpool(images)
            output = torch.flatten(output, start_dim=1)
            features.append(output.cpu().numpy())
            labels.append(label.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def visualize_tsne(features, labels, epoch):
    tsne = TSNE(n_components=2, random_state=0)
    features_tsne = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='viridis', s=5)
    plt.colorbar(scatter)
    plt.title(f't-SNE Visualization - Epoch {epoch}')
    plt.show()

def SSL_fit_model(learner, model, optimizer, SSL_epochs, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    print(f"--SSL Part--")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, SSL_epochs, T_mult=2)

    for epoch in range(SSL_epochs):
        start_time = time.time()

        running_loss = 0.0
        model.train()
        for i, images in enumerate(train_loader):
            print(f"Batch {i+1}/{len(train_loader)} processed", end="\r")
            images = images.to(torch.float32).to(device)
            loss = learner(images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        scheduler.step()

        end_time = time.time()
        spend_time = end_time - start_time
        print(f'Epoch: {epoch+1}/{SSL_epochs} |',
              f'Train_loss: {train_loss:.2f}',
              f'Spend_time: {spend_time:.2f}')

    torch.save(model.state_dict(), "part1/PretrainBYOL.pt")

def FT_fit_model(model, criterion, optimizer, epochs, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    print(f"--FT Part--")
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
            torch.save(model.state_dict(), 'part1/C_best_acc.pt')
            best_acc = val_acc

        if epoch == 0 or epoch == epochs - 1:
            features, labels = get_features(model, train_loader, device)
            visualize_tsne(features, labels, epoch + 1)

    return (training_loss, training_acc, test_loss, test_acc)