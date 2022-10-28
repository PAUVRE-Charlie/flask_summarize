import flask
from flask import request, render_template, url_for
import re


import config
from rouge_score import rouge_scorer

from transformers import pipeline

from gensim.summarization.summarizer import summarize as summarize_gensim


app = flask.Flask(__name__)

default_text = "Aiden Hughes, with an address of Balmoral Road, Bangor, posed as a teenage boy before meeting his victim in a Belfast park.\n\
During the meeting, he touched the girl over her clothes. He later admitted the charge against him.\n\
Belfast Crown Court heard Hughes had marriage problems and took to the internet to \"escape the stress\".\n\
A prosecution lawyer told the court that Hughes met the girl on a social networking site, while pretending to be a 14-year-old called Matt Smith.\n\
They began exchanging emails which soon became sexual.\n\
Hughes asked the girl for meetings and later told her he was 20, and not 14 as previously stated, and admitted his name was Aiden.\n\
His victim reported that he made her feel \"a little bit sorry for him\".\n\
A defence solicitor for Hughes said he was a young man who did an extremely stupid thing and was deeply ashamed of his actions.\n\
The judge ruled Hughes be put on the Sex Offenders' Register for ten years and also made him the subject of a ten-year Sexual Offences Prevention Order, disqualifying him from working with children and restricting his use of the internet.\n\
On his release from prison, he will be required to live at an address approved by the authorities.\n\
The judge told Hughes the impact on the then 14-year-old could not be ignored or forgotten and that adults deliberately making contact with young children for sexual activity would not be tolerated."

default_reference = "A 30-year-old man, who sexually assaulted a 14-year-old girl he met online, has been jailed for 12 months."

# model_size = 't5-small'
# model = T5ForConditionalGeneration.from_pretrained(model_size)
# tokenizer = T5Tokenizer.from_pretrained(model_size)

summarizer_ours = pipeline("summarization", model="CharlieP/t5-small-nlpfinalproject-xsum")
summarizer_facebook = pipeline("summarization", model="facebook/bart-large-cnn")
summarizer_google = pipeline("summarization", model="google/pegasus-xsum")
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def summarize(summarizer, text):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
    return summary

def evaluate_with_rouge(summary, reference):
    return scorer.score(summary, reference)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form.get('text')
        summary = summarize(summarizer_ours, text)
        return render_template("index.html", text=text, summary=summary)
    return render_template("index.html", text=default_text)

@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    if request.method == 'POST':
        text = request.form.get('text')
        reference = request.form.get('reference')
        summary = summarize(summarizer_ours, text)
        scores = evaluate_with_rouge(summary, reference)
        return render_template("evaluate.html", text=text, reference=reference, summary=summary, scores=scores)
    return render_template("evaluate.html", reference=default_reference, text=default_text)


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if request.method == 'POST':
        text = request.form.get('text')
        reference = request.form.get('reference')
        results = {"Our model": {"model": summarizer_ours}, "facebook/bart-large-cnn": {"model": summarizer_facebook}, "google/pegasus-xsum": {"model": summarizer_google}}
        for method in results.keys():
            summarizer = results[method]["model"]
            summary = summarize(summarizer, text)
            scores = evaluate_with_rouge(summary, reference)
            results[method]["summary"] = summary
            results[method]["scores"] = scores
        results["gensim"] = {}
        print(text)
        summary = summarize_gensim(text, word_count=20)
        scores = evaluate_with_rouge(summary, reference)
        results["gensim"]["summary"] = summary
        results["gensim"]["scores"] = scores
        return render_template("compare.html", text=text, reference=reference, results=results)
    return render_template("compare.html", reference=default_reference, text=default_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)