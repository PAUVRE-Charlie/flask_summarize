import json
import flask
from flask import request, render_template, url_for

import config
from rouge_score import rouge_scorer

from transformers import pipeline

app = flask.Flask(__name__)

# model_size = 't5-small'
# model = T5ForConditionalGeneration.from_pretrained(model_size)
# tokenizer = T5Tokenizer.from_pretrained(model_size)

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
        summary = summarize(summarizer_google, text)
        return render_template("index.html", text=text, summary=summary)
    return render_template("index.html", text="The 48-year-old former Arsenal goalkeeper played for the Royals for four years.\
He was appointed youth academy director in 2000 and has been director of football since 2003.\
A West Brom statement said: \"He played a key role in the Championship club twice winning promotion to the Premier League in 2006 and 2012.\""
    )

@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    if request.method == 'POST':
        text = request.form.get('text')
        reference = request.form.get('reference')
        print(summarizer_google(text, max_length=130, min_length=30, do_sample=False)[0]["summary_text"])
        summary = summarize(summarizer_google, text)
        print(summary)
        print(reference)
        scores = evaluate_with_rouge(summary, reference)
        print("SCORES")
        print(scores)
        print(type(scores))
        return render_template("evaluate.html", text=text, reference=reference, summary=summary, scores=scores)
    return render_template("evaluate.html",
        reference="West Brom have appointed Nicky Hammond as technical director, ending his 20-year association with Reading.",
        text="The 48-year-old former Arsenal goalkeeper played for the Royals for four years.\
He was appointed youth academy director in 2000 and has been director of football since 2003.\
A West Brom statement said: \"He played a key role in the Championship club twice winning promotion to the Premier League in 2006 and 2012.\""
    )


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if request.method == 'POST':
        text = request.form.get('text')
        reference = request.form.get('reference')
        results = {"summarizer_facebook": {"model": summarizer_facebook}, "summarizer_google": {"model": summarizer_google}}
        for method in results.keys():
            summarizer = results[method]["model"]
            summary = summarize(summarizer, text)
            scores = evaluate_with_rouge(summary, reference)
            results[method]["summary"] = summary
            results[method]["scores"] = scores
        print(results)
        return render_template("compare.html", text=text, reference=reference, results=results)
    return render_template("compare.html",
        reference="West Brom have appointed Nicky Hammond as technical director, ending his 20-year association with Reading.",
        text="The 48-year-old former Arsenal goalkeeper played for the Royals for four years.\
He was appointed youth academy director in 2000 and has been director of football since 2003.\
A West Brom statement said: \"He played a key role in the Championship club twice winning promotion to the Premier League in 2006 and 2012.\""
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)