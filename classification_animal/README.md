# Classification animal

# Train

## Preprocess Data

- same as count coin
- include normalization + rescale + ToTensor

## Build Model

- need to know what is the output
    - binary classification -> update linear to 1
    - multiple classification -> update linear to number of label

- loss function -> use CrossEntropyLoss
- update getitem -> updaet torch.tensor

# Dataset

https://www.kaggle.com/datasets/alessiocorrado99/animals10