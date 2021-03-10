import torch

from PIL import Image
from torchvision import transforms

class feature_CNN:
    def __init__(self):
        self.model = torch.load("./MNV2_1.pt")#torch.hub.load('pytorch/vision:v0.7.0', 'mobilenet_v2', pretrained=True)
        self.model.eval()

    # sample execution (requires torchvision)
    def getFeatures(self, input):
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        print(input_batch)

        with torch.no_grad():
            # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
            return self.model(input_batch)




if __name__ == '__main__':
    # Download an example image from the pytorch website
    input_image = Image.open("dog.jpg")

    

    CNN = feature_CNN()
    output = CNN.getFeatures(input_image)

    #print(output)
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    #print(probabilities)

    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())