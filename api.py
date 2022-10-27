import flask
from flask import request, render_template, url_for

import config
from rouge_score import rouge_scorer

from transformers import pipeline

app = flask.Flask(__name__)

default_text = "The move is in response to an £8m cut in the subsidy received from the Department of Employment and Learning (DEL).\
The cut in undergraduate places will come into effect from September 2015.\
Job losses will be among both academic and non-academic staff and Queen's says no compulsory redundancies should be required.\
There are currently around 17,000 full-time undergraduate and postgraduate students at the university, and around 3,800 staff.\
Queen's has a current intake of around 4,500 undergraduates per year.\
The university aims to reduce the number of student places by 1,010 over the next three years.\
The BBC understands that there are no immediate plans to close departments or courses, but that the cuts in funding may put some departments and courses at risk.\
The Education Minister Stephen Farry said he recognised that some students might now choose to study in other areas of the UK because of the cuts facing Northern Ireland's universities.\
\"Some people will now be forced to look to opportunities in other parts of Great Britain and may not return to our economy,\" he said.\
\"Defunding our investment in skills, particularly at a time when we're trying to grow the economy does not make a lot of sense. What's happening is we're going backwards.\
\"The loss of any place is damaging to our economy, all subjects teach our young people critical skills.\"\
Queen's vice-chancellor Patrick Johnston said the cuts had the potential to damage the reputation of the university.\
\"The potential negative impact, not just on the university but on the local economy is very significant,\" he said.\
\"It's the last thing we want to do, but we have to begin to focus on those areas where we can grow the organisation and develop it - it's clear we can no longer depend on the public purse to fund tuition.\
\"If we're not competitive we will not attract the best students, and we will not attract the best staff.\"\
Just under £100m, a third of the university's income, comes from the Northern Ireland Executive.\
DEL's budget was reduced by £62m earlier this year, and its budget for higher education institutions fell from £203m to £186m, a reduction of 8.2%.\
Ulster University announced in February that it was dropping 53 courses.\
It will be cutting jobs and student places, but it has not yet revealed how many."

default_reference = "Queen's University Belfast is cutting 236 jobs and 290 student places due to a funding reduction."

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
        summary = summarize(summarizer_google, text)
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
        return render_template("compare.html", text=text, reference=reference, results=results)
    return render_template("compare.html", reference=default_reference, text=default_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)