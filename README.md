# Recipe for vATD Challenge 2025
Please visit the [challenge website](https://www.ncmmsc.org.cn/tsyt/) for more information about the Challenge.

## Data 
The VCTK-RVA dataset is employed in our work, wherein the publicly available VCTK database was annotated for timbre intensity.

The `Dataset/attribute/train.txt` contains the annotation information for the training set of this competition.
The annotation information for some of the data is as follows:
```
明亮_F: p314|p268, p262|p257, p294|p250...
粗_F: p333|p269, p253|p225, p303|p295...
明亮_M: p363|p334, p364|p263, p263|p232...
单薄_M: p285|p363, p226|p298, p275|p364...
...
```
`attribute_gender: speaker A|speaker B` represents the comparison of the strength of the two speakers of the same gender(`F OR M`) in this `attribute`. In the annotation, the speaker on the right, `speakerB`, is stronger in this `attribute` compared to the speaker on the left, `speakerA`.


## Dependencies
Our experiment was conducted in a `Python==3.8.20` environment for training and testing.
Please run the following command :
```
pip install torch==1.12.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
Then run the following command to install other dependencies:
```
pip install -r requirements.txt
```
## 1.Extract speaker embeddings
In this experiment, we used `ECAPA-TDNN` and `FACodec` as pre-trained speaker encoder to extract speaker embeddings.

### 1.1 ECAPA-TDNN
When using `ECAPA-TDNN` as a speaker encoder to extract speaker embeddings, we need to ensure that we have the following two configuration files.
```
ecapa/ecapa/adversary_generation_enc_dec.yaml
ecapa/ecapa/final.params
```
Then, please run the following command.
```
python ecapa/infer_ecapa_emb.py
```
### 1.2 FACodec
When using `FACodec` as a speaker encoder to extract speaker embeddings, we need to ensure that we have the following two configuration files.
```
facodec/checkpoint/ns3_facodec_decoder.bin
facodec/checkpoint/ns3_facodec_encoder.bin
```
Then, please run the following command.
```
python facodec/infer_timbre_emb.py
```

## 2.Train Diff-Net
We train the Diff-Net model based on the extracted speaker embeddings.

When using `ECAPA-TDNN` as a speaker encoder, please run the following command to train the `Diff-Net` model.
```
python train_ecapa.py
```
When using `FACodec` as a speaker encoder, please run the following command to train the `Diff-Net` model.
```
python train_facodec.py
```

## 3.Inference and test
### 3.1 Inference
Perform inference and testing on the two models trained in `2.Train Diff-Net`.

When using `ECAPA-TDNN` as a speaker encoder, please run the following command to train the `Diff-Net` model.
```
python infer_ecapa.py
```
When using `FACodec` as a speaker encoder, please run the following command to train the `Diff-Net` model.
```
python infer_facodec.py
```
After running the `infer` code, we will obtain the discriminative score and predicted labels from Diff-Net.
You can also use the two pre-trained Diff-Net models we have trained to perform inference.
### 3.2 Test
Please run the following command to calculate the `ACC` and `EER` based on the model output results and the true labels.
```
python acc_eer.py
```

## 4.Download of data and model checkpoint
Participants who sign up for the competition can receive the training data and checkpoint.

