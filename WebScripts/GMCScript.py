import sys
if len(sys.argv) < 3:
    print("Error: no input")
    exit()

#needed for I/O
import speech_recognition as sr
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import sys
import wave
import io
from pytube import YouTube


#needed for classifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

#needed for reccomendation
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

model = joblib.load('model.pkl')

ctx_train = model[0]
ctx_count_vect = CountVectorizer()
ctx_x_train_counts = ctx_count_vect.fit_transform(ctx_train[0])
ctx_tfidf_transformer = TfidfTransformer()
ctx_x_train_tfidf = ctx_tfidf_transformer.fit_transform(ctx_x_train_counts)
ctx_clf = MultinomialNB().fit(ctx_x_train_tfidf, ctx_train[1])


cor_train = model[1]
cor_count_vect = CountVectorizer()
cor_x_train_counts = cor_count_vect.fit_transform(cor_train[0])
cor_tfidf_transformer = TfidfTransformer()
cor_x_train_tfidf = cor_tfidf_transformer.fit_transform(cor_x_train_counts)
cor_clf = MultinomialNB().fit(cor_x_train_tfidf, cor_train[1])

grow_train = model[2]
grow_count_vect = CountVectorizer()
grow_x_train_counts = grow_count_vect.fit_transform(grow_train[0])
grow_tfidf_transformer = TfidfTransformer()
grow_x_train_tfidf = grow_tfidf_transformer.fit_transform(grow_x_train_counts)
grow_clf = MultinomialNB().fit(grow_x_train_tfidf, grow_train[1])

def classify(text):
    return {'text':text,
        'context':ctx_clf.predict(ctx_count_vect.transform([text]))[0],
           'correctness':cor_clf.predict(cor_count_vect.transform([text]))[0],
            'growth':grow_clf.predict(grow_count_vect.transform([text]))[0]}

sw = nltk.corpus.stopwords.words('english')
def get_cosine_dist(A, B):
    a_tokens = word_tokenize(A)
    
    b_tokens = word_tokenize(B)
    l1 =[];l2 =[]
    a_set = {w for w in a_tokens if not w in sw} 
    b_set = {w for w in b_tokens if not w in sw}
    rvector = a_set.union(b_set)
    for w in rvector:
        if w in a_set: l1.append(1) 
        else: l1.append(0)
        if w in b_set: l2.append(1)
        else: l2.append(0)
    c = 0
    for i in range(len(rvector)):
        c+= l1[i]*l2[i]
    return c / float((sum(l1)*sum(l2))**0.5)

df = pd.read_csv('../TextData/phrases.csv')


def get_rec(phrase, classification):
    filtered = df.loc[df['context'] == classification['context']]
    filtered = filtered.loc[df['correctness'] == classification['correctness']]
    temp = filtered['text'].apply(lambda x: get_cosine_dist(x, B=phrase))
    filtered.insert(loc=filtered.shape[1],column='cosine_dist',value=temp)
    rec = filtered.sort_values(['growth','cosine_dist'], ascending=(False, False)).head(1)
    rec_text = rec['text'].item()
    rec_dist = rec['cosine_dist'].item()
    return [rec_text, rec_dist]

def get_text(file):
    r = sr.Recognizer()
    with sr.AudioFile(file) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError as ve:
        pass
        #print("Warn: unrecognizable phrase")
    except sr.RequestError as re:
        pass
        #print("Err: request error")

def evaluate_phrase(phrase):
    data = classify(phrase)
    #data['id'] = hash(phrase)
    if data['growth'] == False :
        rec =get_rec("INPUT_DATA", data)
        data['reccomendation'] = rec[0]
        data['similarity'] = rec[1]
    else:
        data['reccomendation'] = "Note: no reccomendation needed"
        data['similarity'] = 0
    return data

CACHE = {}
#minimum silence duration to detect end of phrase in ms
MIN_SILENCE_LEN=1000
#the upper bound for how quiet is silent in dFBS
SILENCE_THRESH=-32
#silence added to the new audio segment in ms
KEEP_SILENCE=500
#the maximim seconds of audio processed at a time (text-to-speech API cannot handle long audio files)
SPLIT_ON= 50
def get_lines(fileName):
    SPLIT_ON = 10
    seg = AudioSegment.from_wav(fileName)
    chunks = split_on_silence(seg, min_silence_len=MIN_SILENCE_LEN, silence_thresh=SILENCE_THRESH, keep_silence=KEEP_SILENCE)
    lines = []
    for i, chunk in enumerate(chunks):
        x = io.BytesIO()
        chunk.export(x, format='wav')
        wav = wave.open(x, 'r')
        dur = wav.getnframes() / wav.getframerate()
        text = ''
        if (dur >= SPLIT_ON):
            t = 0
            for j in range(1, int(dur / SPLIT_ON)):
                y = io.BytesIO()
                chunk[t * SPLIT_ON * 1000:j * SPLIT_ON * 1000].export(y, format='wav')
                text += str(get_text(y)) + ' '
                t = j
            y = io.BytesIO()
            chunk[t * SPLIT_ON * 1000:int(dur) * 1000].export(y, format='wav')
            text += str(get_text(y)) + ' '
        else:
            x = io.BytesIO()
            chunk.export(x, format='wav')
            text = str(get_text(x))
        if text != 'None':
            lines.append(evaluate_phrase(text))
    return (lines)

def get_file_fromYT(url):
    AUDIO_FILE_DIRECTORY = "./Audio/"
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    title = video.title
    #print('Downloading', title,'...')
    out_file = video.download(output_path=AUDIO_FILE_DIRECTORY)
    base, ext = os.path.splitext(out_file)
    base = base.replace(" ", "_")
    if not os.path.exists(base+'.wav'):
        new_file = base + '.mp3'
        print(new_file)
        os.rename(out_file, new_file)
        sound = AudioSegment.from_file(base+'.mp3')
        sound.export(base+'.wav', format="wav")
        os.remove(base+'.mp3')
    #print('Video downloaded.')
    return base+'.wav'



def evaluate_wav(wav_file):
    data = get_lines(wav_file)
    return data

def evaluate_yt(yt_url):
    data = []
    return data

INPUT_TYPE = sys.argv[1]
INPUT_DATA = sys.argv[2]

if INPUT_TYPE == "TEXT":
    print([evaluate_phrase(INPUT_DATA)])
    exit()
elif INPUT_TYPE == "WAV":
    print(evaluate_wav(INPUT_DATA))
    exit()
elif INPUT_TYPE =="YT":
    print(evaluate_wav(get_file_fromYT(INPUT_DATA)))
    exit()
else:
    print("Error: invalid input type")
