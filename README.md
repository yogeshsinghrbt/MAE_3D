# 3D Representation learning for interpolation with masked autoencoder

![Arch](wide.png)

A framework for self-supervised training of vision transformers for interpolation by masking large portions (75%) of the volume, the model is tasked to reconstruct raw pixel values.

Encoder of the model is used to encode the visual patches from sliced frames. The encoded patches are then concatenated with mask tokens which the decoder takes as input. Each Mask token is a shared learned vector that indicates the presence of a missing space to be predicted.

Fixed frame level sin/cos embeddings are added both the input of the encoder and the decoder.

Highlights:
- Customized masking for interpolation that increases performance and reduces information leakage.
- Joint space attention, all the tokens across frames interact with each other in the multihead self-attention layer. 

We are able to learn effective model with only ~60 volumes, which has significant practical value for scenarios with limited data available.

## Dataset
https://thinkonward.com/app/c/challenges/patch-the-planet

## Data Prearation
Please follow the instructions in DATASET.md for data preparation.

## Results

![input](result.gif)

## Acknowledgements
This project is built upon [MAE-pytorch](https://github.com/facebookresearch/mae), [VideoMAE](https://github.com/MCG-NJU/VideoMAE) and [SeismicFoundationModel](https://github.com/shenghanlin/SeismicFoundationModel)
## License

The majority of this project is released under the CC-BY-NC 4.0. Portions of the project are available under separate license terms: pytorch-image-models is licensed under the Apache 2.0 license.
