from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import torch

model_name = "facebook/blenderbot-400M-distill"

def load_model():

    """
        Loads the tokenizer, and model and then return into app.py
    """

    print("Loading Model....Please wait.\n -------------------------")

    #Load Tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #Load Model

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
 
    # Set Device ( CPU or GPU ) 
    # Cpu is used for single processor to work on it like 1 worker has done alot of things
    # GPU is used for 100 of processor to work on it to complete the work

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("\n----------\n")
    print(f"Model loaded on :{device}")

    return tokenizer,model

