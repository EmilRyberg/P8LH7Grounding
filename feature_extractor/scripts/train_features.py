import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_extractor.model import FeatureExtractorNet
from feature_extractor.datasets import TripletDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from torch.utils.tensorboard import SummaryWriter
import random
from feature_extractor.image_utils import unnormalize_image
from tqdm import tqdm


def train_triplet(dataset_dir, weights_dir=None, run_name="run1", epochs=10, on_gpu=True,
                  checkpoint_dir="checkpoints_triplet", batch_size=150, num_features=3, equal_number_of_images_per_class=False, margin=1):
    writer = SummaryWriter(f"runs/triplet_{run_name}")
    dataset = TripletDataset(dataset_dir, "dataset.json", equal_number_of_images_per_class=equal_number_of_images_per_class)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)

    model = FeatureExtractorNet(use_classifier=False, num_features=num_features)
    if weights_dir:
        print(f"Continuing training using weights {weights_dir}")
        model.load_state_dict(torch.load(weights_dir))
        
    example_input = None
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for data in dataloader:
        _, example_input, _ = data
        break
    writer.add_graph(model, example_input)
    # for index, child in enumerate(model.backbone.children()):
    #     if index >= 15:
    #         for param in child.parameters():
    #             param.requires_grad = True
    #     else:
    #         for param in child.parameters():
    #             param.requires_grad = False
    for param in model.backbone.parameters():
       param.requires_grad = False

    if on_gpu:
        model = model.cuda()

    criterion = nn.TripletMarginLoss(margin=margin)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25, weight_decay=0.001, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00)

    running_loss = 0.0
    mini_batches = 0
    epoch_loss = 0.0
    epoch_mini_batches = 0
    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(tqdm(dataloader)):
            class_ids, anchor, positive = data
            if on_gpu:
                anchor = anchor.cuda()
                positive = positive.cuda()

            # First we run the anchors and positives through the network to get embeddings
            optimizer.zero_grad()
            anchor_embeddings = model(anchor)
            anchor_embeddings = F.normalize(anchor_embeddings, p=2) # L2 normalization so embeddings live inside unit hyper-sphere
            positive_embeddings = model(positive)
            positive_embeddings = F.normalize(positive_embeddings, p=2)

            # resample triplets based on embeddings to train on the hardest negative embeddings
            class_ids_np = class_ids.detach().cpu().data.numpy()
            anchor_embeddings_np = anchor_embeddings.detach().cpu().data.numpy()
            positive_embeddings_np = positive_embeddings.detach().cpu().data.numpy()
            all_images = torch.cat((anchor, positive), dim=0)
            # call resample triplets to obtain negative triples, where are the ones violating the margin -> the hardest
            negative_indices = resample_triplets(class_ids_np, anchor_embeddings_np, positive_embeddings_np, alpha=margin)
            negative_indices_tensor = torch.tensor(negative_indices)
            # we run the negative images through
            if on_gpu:
                negative_indices_tensor = negative_indices_tensor.cuda()
            new_negatives = all_images.detach().index_select(0, negative_indices_tensor)
            if on_gpu:
                new_negatives = new_negatives.cuda()
            new_negative_embeddings = model(new_negatives)
            new_negative_embeddings = F.normalize(new_negative_embeddings, p=2)

            if i == len(dataloader) - 2 or (epoch == 0 and i == 0):
                stacked_images = torch.cat((anchor.detach().cpu(), positive.detach().cpu()), dim=0)
                unnormalized_images = None
                for batch_i, batch_img in enumerate(stacked_images):
                    unnormalized_image = unnormalize_image(batch_img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                    if batch_i == 0:
                        unnormalized_images = unnormalized_image.unsqueeze(0)
                    else:
                        unnormalized_images = torch.cat((unnormalized_images, unnormalized_image.unsqueeze(0)), dim=0)
                stacked_embeddings = torch.cat((anchor_embeddings.detach().cpu(), positive_embeddings.detach().cpu()), dim=0)
                meta = []
                for ii, emb in enumerate(stacked_embeddings):
                    meta.append([ii, str(emb.data.numpy())])
                writer.add_embedding(stacked_embeddings, label_img=unnormalized_images, metadata=meta, metadata_header=["index", "embedding"], global_step=(0 if epoch == 0 and i == 0 else epoch+1)) #, global_step=mini_batches

            # compute the triplet loss
            loss = criterion(anchor_embeddings, positive_embeddings, new_negative_embeddings)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item()

            avg_loss = running_loss
            #print(f"[{epoch + 1}, {i + 1}] loss: {avg_loss:.10f}")
            running_loss = 0.0
            mini_batches += 1
            epoch_mini_batches += 1
        avg_epoch_loss = epoch_loss / epoch_mini_batches
        epoch_mini_batches = 0
        print(f"[{epoch + 1}] loss: {avg_epoch_loss:.10f}")
        writer.add_scalar("training loss", avg_epoch_loss, mini_batches)

        checkpoint_name = f"triplet-epoch-{epoch + 1}-loss-{avg_epoch_loss:.5f}.pth"
        checkpoint_full_name = os.path.join(checkpoint_dir, checkpoint_name)
        print(f"[{epoch + 1}] Saving checkpoint as {checkpoint_full_name}")
        torch.save(model.state_dict(), checkpoint_full_name)
        dataset.sample_pairs()  # get new triplets
        epoch_loss = 0
    print("Finished training")
    writer.close()


# this function resamples the triplets from anchors and positive embeddings, in order to obtain the best negative samples
def resample_triplets(class_id_np, anchor_emb, positive_emb, alpha=0.2):
    new_negative_indices = []
    class_id_np_2x = np.tile(np.expand_dims(class_id_np, axis=1), (2, 1)) # just copies the class_id list so it is twice as long
    all_embeddings_np = np.concatenate((anchor_emb, positive_emb), axis=0) # concatenates anchor and positive embeddings
    for i, (cid, a_emb, p_emb) in enumerate(zip(class_id_np, anchor_emb, positive_emb)):
        pos_dist = np.linalg.norm(a_emb - p_emb) # compute the distance between the anchor and positive embedding
        original_indices = get_all_other_images_and_embeddings(class_id_np_2x, cid) # gets the indices of all other classes
        a_repeat_emb = np.tile(a_emb, (len(original_indices), 1)) # repeats the anchor embedding to be length of the original_indices
        negative_emb = np.take(all_embeddings_np, original_indices, axis=0) # take all the embeddings with the indicies in the original_indices list
        neg_dist = np.linalg.norm(a_repeat_emb - negative_emb, axis=1).reshape((-1, 1)) # compute the distance between the anchor and all the negative embeddings
        pos_dist_repeat = np.tile(pos_dist, (negative_emb.shape[0], 1))
        neg_indices, _ = np.nonzero(neg_dist - pos_dist_repeat < alpha) # the indices which are below alpha to the positive
        if neg_indices.shape[0] == 0:
            batch_len = len(original_indices)
            r_idx = random.randint(0, batch_len-1)
            new_negative_indices.append(original_indices[r_idx])
        else:
            np.random.shuffle(neg_indices)
            neg_index = neg_indices[0]
            new_negative_indices.append(original_indices[neg_index])

    return new_negative_indices


def get_all_other_images_and_embeddings(class_ids_np, current_class_id):
    original_indices = []

    for i, cid in enumerate(class_ids_np):
        if cid != current_class_id:
            original_indices.append(i)
    return original_indices


if __name__ == '__main__':
    train_triplet(dataset_dir="dataset_2_even_more_filtered", run_name="run2_d2_even_more_filtered", checkpoint_dir="checkpoints_triplet2_d2_even_more_filtered", weights_dir="checkpoints1_d2_even_more_filtered_3emb/epoch-85-loss-0.01860-99.67.pth", num_features=3, batch_size=200,
                  epochs=30, equal_number_of_images_per_class=True, margin=1)