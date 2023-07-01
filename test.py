#testing the model with separate dataset of test.csv where you can view .
#with fine-tuning testing model with test csv dataset

#we already loaded test.csv file  

test=tweet(test_path,token,max_length,is_test=True)
load=DataLoader(test,batch_size=1,shuffle=False)

model.eval()
predictions=[]


#predictions
from tqdm import tqdm
with torch.no_grad():
    for batch in tqdm( load):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        batch_predictions = torch.argmax(logits, dim=1).tolist() # Get the predicted class index
        predictions.extend(batch_predictions)

# Print the predictions
#print(predictions)


#without fine-tuning

predictions=model.predict(test_dataset2.batch(32))
