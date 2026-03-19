from model_loader import load_model
import torch

#Load the tokenizer, and model

tokenizer,model = load_model()

print("\n AI Chatbot Ready! Type 'exit' to stop.\n")

conversation_history=[]

while True:
    user_input = input("You:   ")

    #Exit Condition
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break
    
    #store user input
    conversation_history.append(user_input)

    #keep all the messages
    context = " ".join(conversation_history[:])

    #Tokenize input and move to same devices as model
    inputs = tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(model.device)

    #Generate response
    with torch.no_grad():  #optimization(no training)
        output = model.generate(  #reply_ids for users
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_k=60,  #Top 60 words only show on it
            top_p = 0.95,    #accuracy level for all the answers
            do_sample=True
        )
    
    #Decode responses i.e. Tokenizers are numbers convert into text

    reply = tokenizer.decode(output[0],skip_special_tokens=True)

    print("Bot: ",reply)

    conversation_history.append(reply)
