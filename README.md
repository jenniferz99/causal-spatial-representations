# LLM Causal Spatial Representations

This is our official code base for the paper _More than Correlation: Do Large Language Models Learn Causal Representations of Space?_

Authors: Yida Chen, Yixian Gan, Sijia Li, Li Yao, Xiaohan Zhao (listed in alphabetical order of the last names)


## How to Use?
### intervention.ipynb
This notebook includes code for **probes training and intervention for downstream country prediction task, as well as RSA (Section 4, 5 and 6.1 in the paper)**. Users should be able to run it sequntially. Just set the variable `path` to the location of the submission folder. 

```python
# In Prepare dataset section

# Change this to your directory 
path = '/content/drive/MyDrive/Colab Notebooks/6.8610_proj' 

df = pd.read_csv(os.path.join(path, 'majorcities.csv'), index_col=0).dropna(subset='population')
...
```

This notebook is organized as follows: 
- Prepare train/val/test datasets 
- Utility functions (e.g. load pretrained LLM) 
- Train linear probe
- Train multi-layer FFNN probes
- Intervention on downstream classificaition tasks
- Intervention significance test
- RSA

We already extracted the activation from intermediate layers of BERT and GPT, which can be loaded by 

```python
# Select LM 
model_name = 'GPT-Neo_sent'
# model_name = 'DeBERTa_sent'
mask = torch.load(os.path.join(path, f'hidden_states/{model_name}/masks.pt'))
emb = torch.load(os.path.join(path, f'hidden_states/{model_name}/h24.pt')) # Change this line if want to use activations from other later
id = torch.load(os.path.join(path, f'hidden_states/{model_name}/cid.pt'))
token = torch.load(os.path.join(path, f'hidden_states/{model_name}/tokens.pt'))
d = {id[i].item():(emb[i], token[i], mask[i]) for i in range(id.shape[0])}
```
*Note:* Each `h*.pt` file is about 2Gb. For storage capacity concern, we *only provides activations from layer 0, 6, 16, 24 of GPT*. Activations from all hidden layers of BERT are avaliable. All data are avaliable [here](https://drive.google.com/drive/u/1/folders/185Vv_Kfx7rg5c3kv0P5BkOI78t4ctpnf).


`load_bert(layer)` and `load_gpt(layer)` are functions used to load pretrained LLMs. The argument `layer` specifies the index of the hidden transformer layer whose activation we would intervent. E.g. if we want to do intervention on the activation from the 12th layer of GPT-Neo, then load the model by 
```python
tokenizer, model = load_gpt(12)
```

We provided a trained classification head layer for country classification to save some time, which can be loaded by 
```python
tokenizer, model = load_gpt(12)
clf = Classifier(100,in_features=2048, n_layer=1, hidden_dim=2, transformer=model, device=device)
# Load pretrained classification layer 
clf.linear = torch.load(os.path.join(path, 'gpt_cls_lin.pth'),map_location=device)
```

Training and validation dataset can be loaded by 
```python
train_loader, val_loader = load_data(d, bs=256, feat_name='hidden_state')
```
where `bs` is the batch size. `feat_name` needs to be changed to `last_hidden_state` if users want to use the activation from the last hidden layer of LLM. 


### intervention_nextword.ipynb
This notebook contains code for **causal intervention on LLM next word prediction (Section 6.2 in the paper)**. See in-line comments for detailed instructions on how to run the code.