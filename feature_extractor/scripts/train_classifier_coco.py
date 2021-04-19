from feature_extractor.datasets import ClassificationDataset
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
import os
from torchvision.models.mobilenet import mobilenet_v2
from collections import OrderedDict
import math
from tqdm import tqdm
from feature_extractor.image_utils import unnormalize_image
from feature_extractor.model import FeatureExtractorNet


# function for first step in training, train a classifier
def train_softmax(dataset_dir, weights_dir=None, run_name="run1", epochs=80, continue_epoch=None,
                  on_gpu=True, checkpoint_dir="checkpoints", batch_size=64, print_interval=50, num_features=3):
    writer = SummaryWriter(f"runs/{run_name}")

    if dataset_dir[-1] != '/':
            dataset_dir += '/'
    dataset = ClassificationDataset(dataset_dir, "annotations/instances_train2017.json", image_folder_path="train2017")
    dataset_length = len(dataset)
    val_dataset = ClassificationDataset(dataset_dir, "annotations/instances_val2017.json", image_folder_path="val2017")
    val_dataset_length = len(val_dataset)

    train_length = int(math.ceil(dataset_length * 0.3))
    test_length = dataset_length - train_length
    train_set, _ = torch.utils.data.random_split(dataset, [train_length, test_length])

    dataloader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=2)

    num_classes = dataset.get_num_classes()
    print(f"Number of classes: {num_classes}")
    model = FeatureExtractorNet(use_classifier=True, num_features=num_features, num_classes=num_classes)

    for index, child in enumerate(model.backbone.children()):
        if index >= 15: # This will make the last 4 layers trainable
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

    example_input = None
    for data in dataloader:
        images, labels = data
        example_input = images
        break

    writer.add_graph(model, example_input)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01) # lr 0.001 default

    print(f"Training with {train_length} train images, and {val_dataset_length} test images")

    if on_gpu:
        model = model.cuda()
    
    running_loss = 0.0

    # here we start training
    got_examples = False
    start = 0 if continue_epoch is None else continue_epoch
    for epoch in range(start, epochs):
        model.train()
        for i, data in enumerate(tqdm(dataloader)):
            inputs, labels = data
            if not got_examples:
                unnormalized_inputs = None #unnormalize_image(inputs, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                for batch_i, batch_img in enumerate(inputs):
                    unnormalized_image = unnormalize_image(batch_img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                    if batch_i == 0:
                        unnormalized_inputs = unnormalized_image.unsqueeze(0)
                    else:
                        unnormalized_inputs = torch.cat((unnormalized_inputs, unnormalized_image.unsqueeze(0)), dim=0)
                grid = torchvision.utils.make_grid(unnormalized_inputs)
                got_examples = True
                writer.add_image("images", grid, 0)

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
        #print("Testing")

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
                softmax_output = F.softmax(outputs, dim=0)
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
        test_acc = (test_correct / val_dataset_length) * 100.0

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


if __name__ == "__main__":
    train_softmax(dataset_dir="F:/Projects/P8/coco/", run_name="run2_coco", continue_epoch=None, print_interval=20,
                  checkpoint_dir="checkpoints2_coco", #weights_dir="checkpoints4_3emb/epoch-45-loss-0.19246-90.92.pth",
                  num_features=128, on_gpu=True, epochs=10)