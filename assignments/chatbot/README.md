<<<<<<<
A neural chatbot using sequence to sequence model with
attentional decoder. This is a fully functional chatbot.

This is based on Google Translate Tensorflow model 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen as the starter code for assignment 3,
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

The detailed assignment handout and information on training time can be found at http://web.stanford.edu/class/cs20si/assignments/a3.pdf 

<h2>Sample conversation</h2>
HUMAN ++++ hi <br>
BOT ++++ hi . what ' s your name ?<br>
HUMAN ++++ i'm chip<br>
BOT ++++ please .<br>
HUMAN ++++ please what?<br>
BOT ++++ i don ' t know what to say .<br>
HUMAN ++++ where are you from?<br>
BOT ++++ california .<br>
HUMAN ++++ do you like california?<br>
BOT ++++ i ' m okay .<br>
HUMAN ++++ i like california a lot<br>
BOT ++++ of what ?<br>
HUMAN ++++ i live here<br>
BOT ++++ and what is it ?<br>
HUMAN ++++ california<br>
BOT ++++ i don ' t know what to say .<br>
HUMAN ++++ i'm a student here<br>
BOT ++++ and get it out of here .<br>
HUMAN ++++ what do you like to do in your free time?<br>
BOT ++++ and what about the gun ?<br>
HUMAN ++++ do you have a gun?<br>
BOT ++++ yes .<br>
HUMAN ++++ you're dangerous!<br>
BOT ++++ i ' m not .<br>
HUMAN ++++ do you go hunting?<br>
BOT ++++ i can do .<br>
HUMAN ++++ what do you hunt?<br>
BOT ++++ virgin .<br>
HUMAN ++++ that's very bad<br>
BOT ++++ and you in love ?<br>
HUMAN ++++ no i'm not<br>
BOT ++++ please - -<br>
HUMAN ++++ okay, i am<br>
BOT ++++ and you are good .<br>
HUMAN ++++ yeah i am good too.<br>
BOT ++++ good good to you .<br>
HUMAN ++++ are you in love?<br>
BOT ++++ i ' m ready .<br>
HUMAN ++++ ready for what?<br>
BOT ++++ for what ?<br>
HUMAN ++++ for what?<br>
BOT ++++ i don ' t know what to say .<br>
HUMAN ++++ tell me<br>
BOT ++++ i can do it .<br>
HUMAN ++++ okay, tell me what you're ready for<br>
BOT ++++ i ' m doing you .<br>
HUMAN ++++ no you're not<br>
BOT ++++ please !<br>
HUMAN ++++ this is a strange conversation<br>
BOT ++++ i don ' t know what to say .<br>
HUMAN ++++ are you a guy?<br>
BOT ++++ of what ?<br>
HUMAN ++++ are you a boy?<br>
BOT ++++ i ' m fine .<br>
HUMAN ++++ are you a man?<br>
BOT ++++ i ' m a little .<br>
HUMAN ++++ what's your name?<br>
BOT ++++ chris -<br>
HUMAN ++++ chris who?<br>
BOT ++++ cornelius .<br>
HUMAN ++++ so i guess you're a guy.<br>
BOT ++++ i am .<br>
HUMAN ++++ okay<br>
BOT ++++ okay ?<br>
HUMAN ++++ do you have a girlfriend?<br>
BOT ++++ i ' m sure .<br>
HUMAN ++++ what's her name?<br>
BOT ++++ let ' s talk about something else .<br>

See output_convo.txt for more sample conversations.

<h2>Usage</h2>

Step 1: create a data folder in your project directory, download
the Cornell Movie-Dialogs Corpus from 
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
Unzip it

Step 2: python data.py
<br>This will do all the pre-processing for the Cornell dataset.

Step 3:
python chatbot.py --mode [train/chat] <br>
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
