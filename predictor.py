import torch
from transformers import AutoTokenizer
from config import *
from model import HeadlineClassifier


class PythonPredictor:
    def __init__(self):
        self.device = device
        checkpoint = torch.load('headline_model.pt',map_location=torch.device(self.device))
        self.model = HeadlineClassifier(model_name,num_classes)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
    def predict(self, input_ids,input_mask_array):
      label,probability,probs = self.model(input_ids.to(self.device),input_mask_array.to(self.device))
      return label,probability.item(),probs

# if __name__ == '__main__':
#   predictor = PythonPredictor()
#   print(predictor.predict('Viral News: অলৌকিক ঘটনা! একই সন্তানের দু-দুবার জন্ম দিলেন মা! নিজেই জানালেন আশ্চর্য কাহিনী'))