import torch
  
# CLS token - one vector per input - commonly used as a representation of the entire sentence or text, so we will use it for each text
def embed_bert_cls(text, model, tokenizer, device_type = 'cuda'):

    valid_status = ['cuda', 'cpu']
    if device_type not in valid_status:
        raise ValueError('choose between cuda and cpu')
    
    if device_type == 'cuda':
        print('cuda is not availvble, switching to cpu')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    
    model.to(device)

    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        global model_output;
        model_output = model(**{k: v.to(model.device) for k, v in t.items()}) # It feeds tokenized input t into the model, after making sure all tensors are on the right device (GPU/CPU), and stores the result
    embeddings = model_output.last_hidden_state[:, 0, :] # It extracts the embedding of the CLS token from the model output
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].to("cpu").numpy() # assume single text input - one vector output