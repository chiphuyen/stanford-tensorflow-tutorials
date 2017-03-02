<<<<<<<
A neural chatbot using sequence to sequence model with
attentional decoder. This is a fully functional chatbot.

This is based on Google Translate Tensorflow model 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen as the starter code for assignment 3,
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

The detailed assignment handout can be found at http://web.stanford.edu/class/cs20si/assignments/a3.pdf

<h2>Sample conversation</h2>
HUMAN ++++ hi
BOT ++++ hi . what ' s your name ?
HUMAN ++++ i'm chip
BOT ++++ please .
HUMAN ++++ please what?
BOT ++++ i don ' t know what to say .
HUMAN ++++ where are you from?
BOT ++++ california .
HUMAN ++++ do you like california?
BOT ++++ i ' m okay .
HUMAN ++++ i like california a lot
BOT ++++ of what ?
HUMAN ++++ i live here
BOT ++++ and what is it ?
HUMAN ++++ california
BOT ++++ i don ' t know what to say .
HUMAN ++++ i'm a student here
BOT ++++ and get it out of here .
HUMAN ++++ what do you like to do in your free time?
BOT ++++ and what about the gun ?
HUMAN ++++ do you have a gun?
BOT ++++ yes .
HUMAN ++++ you're dangerous!
BOT ++++ i ' m not .
HUMAN ++++ do you go hunting?
BOT ++++ i can do .
HUMAN ++++ what do you hunt?
BOT ++++ virgin .
HUMAN ++++ that's very bad
BOT ++++ and you in love ?
HUMAN ++++ no i'm not
BOT ++++ please - -
HUMAN ++++ okay, i am
BOT ++++ and you are good .
HUMAN ++++ yeah i am good too.
BOT ++++ good good to you .
HUMAN ++++ are you in love?
BOT ++++ i ' m ready .
HUMAN ++++ ready for what?
BOT ++++ for what ?
HUMAN ++++ for what?
BOT ++++ i don ' t know what to say .
HUMAN ++++ tell me
BOT ++++ i can do it .
HUMAN ++++ okay, tell me what you're ready for
BOT ++++ i ' m doing you .
HUMAN ++++ no you're not
BOT ++++ please !
HUMAN ++++ this is a strange conversation
BOT ++++ i don ' t know what to say .
HUMAN ++++ are you a guy?
BOT ++++ of what ?
HUMAN ++++ are you a boy?
BOT ++++ i ' m fine .
HUMAN ++++ are you a man?
BOT ++++ i ' m a little .
HUMAN ++++ what's your name?
BOT ++++ chris -
HUMAN ++++ chris who?
BOT ++++ cornelius .
HUMAN ++++ so i guess you're a guy.
BOT ++++ i am .
HUMAN ++++ okay
BOT ++++ okay ?
HUMAN ++++ do you have a girlfriend?
BOT ++++ i ' m sure .
HUMAN ++++ what's her name?
BOT ++++ let ' s talk about something else .

See output_convo.txt for more sample conversations.

<h2>Usage</h2>

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

If you find the tutorial helpful, please head over to <a href="http://web.stanford.edu/class/cs20si/anonymous_chatlog.pdf">Anonymous Chatlog Donation</a>
to see how you can help us create the first realistic dialogue dataset.

Thank you very much!
>>>>>>> origin/master
