# DistilBert-Fine-Tune-
________________________________________________________________________________________________________________________________________________________________________________________

Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

But, it’s not always clear whether a person’s words are actually announcing a disaster. Take this example:

The author explicitly uses the word “ABLAZE” but means it metaphorically. This is clear to a human right away, especially with the visual aid. But it’s less clear to a machine.

 build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t. You’ll have access to a dataset of 10,000 tweets that were hand classified

Certainly! Here are a few examples of natural disaster-related tweets :

    "🚨 Breaking News: A massive earthquake measuring 7.5 on the Richter scale struck the coastal region today. Prayers for the safety of everyone affected. Stay safe and be prepared! 🙏 #Earthquake #SafetyFirst"

    "🔥 Wildfires are spreading rapidly in the forest area, posing a serious threat to nearby communities. Emergency services are on high alert. Evacuation orders have been issued. Please follow instructions from authorities. #Wildfires #SafetyAlert"

    "⚠️ Tropical Storm Alert: The meteorological department has issued a warning for a potential tropical storm formation in the coming days. Stay tuned for updates and take necessary precautions. #TropicalStorm #StaySafe"

    "💨 Strong winds and heavy rainfall are expected in the region due to an approaching cyclone. Secure loose objects, stay indoors, and avoid unnecessary travel. Safety should be the top priority. #Cyclone #WeatherUpdate"

 for solving this we used Bert-base-cased and bert-base-uncased with fine tuning the model and pretrained model from TFBertForSequenceClassification.
 trained the model separate-separate datasets as train and test

 Model: "tf_bert_for_sequence_classification"::
                                             
                                              Total params: 109,483,778 

with fine tuning the model


 Model: "tf_bert_for_sequence_classification"::


                                            Total params: 177,855,747

________________________________________________________________________________________________________________________________________________________________________________________________________

🛠 frameworks and tools used:

<img align="left" alt="Python" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />


<img align="left" alt="Tensorflow" src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" />

<img align="left" alt="Pandas" src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" />


<img align="left" alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black" />

<img align="left" alt="Scikit-learn" src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" />

<img align="left" alt="Gradio" src="https://gradio.app/" />

<img align="left" alt="keras" src="https://keras.io/" />
