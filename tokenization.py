import tiktoken

encoder = tiktoken.encoding_for_model('gpt-4o')
print(encoder.n_vocab)

txt = "the cat sat on the mat"
tokens = encoder.encode(txt)
print(tokens)   


my_tokens =[976, 9059, 10139, 402, 290, 2450]
decoded = encoder.decode( my_tokens )
print ("Decoded : ", decoded)

# GPT: Generative Pre-trained Transformer
"""
 Tokenization            : convert user input into tokens/number
 Vector Embedding        : capture semantic meaning.
 Postional Encoding      : inputs with same words can look same to transformer, so it add's position information.
 Multi-head Attention    : allows token to have conversation, to adjust their embeddings.
 Neural-Net/Feed-Forward : processes these tokens with their embeddings
 Layer Normalization     : Stabilizes training and helps information flow through the network.
 Output Layer            : ( What comes next ) Produces probabilities for the next token, 
                           from which GPT predicts the most likely continuation.

                           
Knowlegde Cutoff : the dataset or record upto the date which it has or has been trained.
Temperature      : increasing increases the randomness or creativity of the output.
Softmax          : chosing high probability output.

by feeding realtime data or context, we can make realtime llm's.
RAG in simple terms is solving realtime user query by providing the real time context to the model from external source.


Interesting Observation:
    Most modern LLMs have been trained on large amounts of human-generated
    internet data.
    As AI-generated content becomes a larger fraction of the internet,
    future training datasets may contain significant amounts of synthetic
    (AI-generated) data.

    Excessive training on low-quality synthetic data can reduce model
    quality, a phenomenon sometimes called "model collapse."

    However, carefully filtered and high-quality synthetic data can
    improve models and is already used successfully by leading AI labs.

Some Types of models (Prompt-Engineering):
    -> Chain of Thought.
    -> Self-Consistency.
    -> Persona-Based.
    -> Role-Playing.

"""

print() 