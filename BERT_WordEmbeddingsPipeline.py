import torch
import numpy as np

# !pip install pytorch-pretrained-bert
from pytorch_pretrained_bert import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
# Put the model in "evaluation" mode, meaning feed-forward operation.
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

def embeddingsPipeline(text):
    
    # Add the special tokens.
    marked_text = "[CLS] " + text + " [SEP]"
    
    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)
    
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    
    # Mark each of the tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)
    
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
        
    """
    encoded_layers object has four dimensions, in the following order:
        1. The layer number (12 layers)
        2. The batch number (1 sentence)
        3. The word / token number (no. of tokens in our sentence) (let it be 'N')
        4. The hidden unit / feature number (768 features)
    """ 
    
    # Concatenate the tensors for all layers.
    # We use `stack` here to create a new dimension in the tensor.
    token_embeddings = torch.stack(encoded_layers, dim=0) 
    # Dimensions of token_embeddings = [12 x 1 x N x 768]
    
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # Dimensions of token_embeddings = [12 x N x 768]
    
    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)
    # Dimensions of token_embeddings = [N x 12 x 768]
    
    """
    Studies for Named Entity Recognition have shown that concatenation of the last four layers 
    produced the best results on this specific task.
    Have a look @ http://jalammar.github.io/images/bert-feature-extraction-contextualized-embeddings.png
    """
    # Creating the word vectors by summing together the last four layers.

    token_vecs_sum = [] 
    # Dimensions of token_vecs_sum will be [N x 768] 
    
    for token in token_embeddings:
        # `token` is a [12 x 768] tensor
        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        
        token_vecs_sum.append(sum_vec)
        
    
    # Converting embeddings into numpy array for further processing.
    numpy_embeddings = []
    for embedding in token_vecs_sum:
        numpy_embeddings.append(embedding.numpy())
    
        
    # Now we have to deal with the split words. 
    # Example: 'embeddings' is tokenized into ['em', '##bed', '##ding', '##s']
    # We will average the embeddings of split words to get the embeddings of the original word. 
    
    # Put all split-word-embeddings (if any) of every original word in a list 'p'. 
    # If the word is not a split word, make a list with one elment i.e, 
    # the respective word-embedding and append this list to 'p'. 
    p = []
    for i,token in enumerate(tokenized_text): 
        if token[0] == '#' and token[1] == '#':
            p[-1].append(numpy_embeddings[i])
        else:
            p.append([numpy_embeddings[i]])
    
    
    # Averaging the embeddings of split words to obtain the embedding of original word.  
    final_embeddings = []
    for i in p:
        temp = np.mean( np.array([l for l in i]), axis=0 )
        final_embeddings.append(temp)
        
    # Remove first and last embeddings (i.e., Embeddings for "[CLS]" and "[SEP]")
    final_embeddings = final_embeddings[1:-1]
    
    return final_embeddings 