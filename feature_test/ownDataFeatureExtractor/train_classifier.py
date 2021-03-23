from datasets import ClassificationDataset
from model import FeatureExtractorNet
import torch
from tensorboardX import SummaryWriter
import os
from collections import OrderedDict
import math

def create_model(num_features=64, num_classes=5):
    # Load pretrained model
    model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)

    # Remove classification FC layer
    classifier_name, old_classifier = model._modules.popitem()

    # Add new feature and classification layer
    classifier_input_size = old_classifier[1].in_features

    classifier = torch.nn.Sequential(OrderedDict([
                            ('fc1', torch.nn.Linear(classifier_input_size, num_features)),
                            ('relu1', torch.nn.ReLU()),
                            ('fc2', torch.nn.Linear(num_features, num_classes)),
                            ]))

    model.add_module(classifier_name, classifier)

    return model


# function for first step in training, train a classifier
def train_softmax(dataset_dir, weights_dir=None, run_name="run1", epochs=30,
                  on_gpu=True, checkpoint_dir="checkpoints", batch_size=100, print_interval=50, num_classes=5):

    writer = SummaryWriter(f"runs/{run_name}")

    if dataset_dir[-1] != '/':
            dataset_dir += '/'
    dataset = ClassificationDataset(dataset_dir, "dataset.json")
    dataset_length = len(dataset)

    train_length = int(math.ceil(dataset_length * 0.7))
    test_length = dataset_length - train_length
    train_set, test_set = torch.utils.data.random_split(dataset, [train_length, test_length])

    dataloader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=0)

    model = create_model() #FeatureExtractorNet(use_classifier=True, num_features=32, num_classes=num_classes)

    count = 0
    for child in model.features.children(): #model.backbone.children(): #19 children, not sure why but rolling with it
        count += 1
        if count >= 15: # This will make the last 4 layers trainable
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    """for param in model.bottleneck.parameters():
        param.requires_grad = True"""


    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if weights_dir:
        print(f"Continuing training using weights {weights_dir}")
        model.load_state_dict(torch.load(weights_dir))

    """example_input = None
    for data in dataloader:
        images, labels = data
        example_input = images
        break"""

    #writer.add_graph(model, example_input)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01) # lr 0.001 default

    print(f"Training with {train_length} train images, and {test_length} test images")

    if on_gpu:
        model = model.cuda()
    
    running_loss = 0.0

    # here we start training
    for epoch in range(epochs):
        print("Training")
        for i, data in enumerate(dataloader, 0):
            model.train()
            inputs, labels = data

            if on_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % print_interval == print_interval-1:
                loss = running_loss / print_interval
                print(f"[{epoch + 1}, {i + 1}] loss: {loss:.6f}")
                running_loss = 0.0
                writer.add_scalar("training loss", loss, epoch * len(dataloader) + i)

        test_loss = 0
        test_correct = 0
        total_test_correct = 0
        total_img = 0
        total_runs = 0
        print("Testing")

        for i, data in enumerate(test_dataloader, 0):
            model.eval()
            with torch.no_grad():
                inputs, labels = data
                if on_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                softmax_output = torch.nn.functional.softmax(outputs, dim=0)
                output_np = softmax_output.cpu().data.numpy()
                predicted_ids = output_np.argmax(1)
                labels_np = labels.cpu().data.numpy()
                correct_labels = labels_np == predicted_ids
                sum_correct_labels = correct_labels.sum()
                test_correct += sum_correct_labels

                total_correct = correct_labels
                total_correct_sum = total_correct.sum()
                total_test_correct += total_correct_sum

                total_runs += 1

        avg_test_loss = test_loss / total_runs
        test_acc = (test_correct / test_length) * 100.0

        print(f"[{epoch + 1}] Test loss: {avg_test_loss:.5f}")
        print(f"[{epoch + 1}] Test acc.: {test_acc:.3f}%")

        writer.add_scalar("test loss", avg_test_loss, epoch + 1)
        writer.add_scalar("test accuracy", test_acc, epoch + 1)

        checkpoint_name = f"epoch-{epoch + 1}-loss-{avg_test_loss:.5f}-{test_acc:.2f}.pth"
        checkpoint_full_name = os.path.join(checkpoint_dir, checkpoint_name)
        print(f"[{epoch + 1}] Saving checkpoint as {checkpoint_full_name}")
        torch.save(model.state_dict(), checkpoint_full_name)

    print("Finished training")
    writer.close()


train_softmax(dataset_dir="./dataset_output/", run_name="run4", checkpoint_dir="checkpoints4", num_classes=5)