<<<<<<< HEAD
A neural chatbot using sequence to sequence model with
attentional decoder. 

This is based on Google Translate Tensorflow model 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen as the starter code for assignment 3,
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

The detailed assignment handout can be found at: (not updated)

See output_convo.txt for sample conversations.

Usage:
Step 1: create a data folder in your project directory, download
the Cornell Movie-Dialogs Corpus from 
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
Unzip it

Step 2: python data.py
This will do all the pre-processing for the Cornell dataset.

Step 3:
python chatbot.py --mode [train/chat]
If mode is train, then you train the chatbot. By default, the model will
restore the previously trained weights (if there is any) and continue
training up on that.

If you want to start training from scratch, please delete all the checkpoints
in the checkpoints folder.

If the mode is chat, you'll go into the interaction mode with the bot.

By default, all the conversations you have with the chatbot will be written
into the file output_convo.txt in the processed folder. If you run this chatbot,
I kindly ask you to send me the output_convo.txt so that I can improve
the chatbot. My email is huyenn@stanford.edu

If you find the tutorial helpful, please head over to Anonymous Chatlog Donation
to see how you can help us create the first realistic dialogue dataset.

Thank you very much!
=======
You can see sample conversations in output_convo.txt <br>
Starter code and assignment handout will be out in a few hours
>>>>>>> origin/master
