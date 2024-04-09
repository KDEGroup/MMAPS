from __future__ import division
import collections
import six
import numpy as np
from .rouge import Rouge

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from bert_score import score

def _ngrams(words, n):
    queue = collections.deque(maxlen=n) # 双端队列
    for w in words:
        queue.append(w)
        if len(queue) == n:
            yield tuple(queue)

def _ngram_counts(words, n):
    return collections.Counter(_ngrams(words, n))

def _ngram_count(words, n):
    return max(len(words) - n + 1, 0)

def _counter_overlap(counter1, counter2):
    result = 0
    for k, v in six.iteritems(counter1):
        result += min(v, counter2[k])
    return result

def _safe_divide(numerator, denominator):
    if denominator > 0:
        return numerator / denominator
    else:
        return 0

def _safe_f1(matches, recall_total, precision_total, alpha=0.5):
    recall_score = _safe_divide(matches, recall_total)
    precision_score = _safe_divide(matches, precision_total)
    denom = (1.0 - alpha) * precision_score + alpha * recall_score
    if denom > 0.0:
        return (precision_score * recall_score) / denom
    else:
        return 0.0

def rouge_n(peer, model, n):
    """
    Compute the ROUGE-N score of a peer with respect to one or more models, for
    a given value of `n`.
    """
    peer_counter = _ngram_counts(peer, n)
    '''
    for model in models:
        model_counter = _ngram_counts(model, n)
        matches += _counter_overlap(peer_counter, model_counter)
        recall_total += _ngram_count(model, n)
    '''
    model_counter = _ngram_counts(model, n)
    matches = _counter_overlap(peer_counter, model_counter)
    recall_total = _ngram_count(model, n)
    precision_total =  _ngram_count(peer, n)
    # print(matches, recall_total, precision_total)
    return _safe_f1(matches, recall_total, precision_total)


def rouge_1_corpus(peers, models):
    curpus_size = len(peers)
    rouge_score = 0
    for (peer, model) in zip(peers, models):
        rouge_score += rouge_n(peer, model, 1)
    return rouge_score / curpus_size


def rouge_2_corpus(peers, models):
    curpus_size = len(peers)
    rouge_score = 0
    for (peer, model) in zip(peers, models):
        rouge_score += rouge_n(peer, model, 2)
    return rouge_score / curpus_size

def srouge(peers, models, device, print_log=print):
    # peers: reference
    # models: candidate
    assert len(peers) == len(models)
    xsrcs, xtgts = [], []
    count_tgts = []
    for s, ts in zip(models, peers):
        count_tgts.append(len(ts))
        for t in ts:
            xsrcs.append(s)
            xtgts.append(t)
    xxsrcs, xxtgts = [], []
    for s, t in zip(xsrcs, xtgts):
        xxsrcs.append(" ".join(list(s)))
        xxtgts.append(" ".join(list(t)))

    scores = Rouge(metrics=["rouge-l"]).get_scores(xxsrcs, xxtgts, avg=False)
    rouge1f_score, rouge2f_score, rougelf_score = [], [], []
    for indx, one in enumerate(scores):
        rouge1f_score.append(rouge_n(xxsrcs[indx].split(), xxtgts[indx].split(), 1))
        rouge2f_score.append(rouge_n(xxsrcs[indx].split(), xxtgts[indx].split(), 2))
        rougelf_score.append(one['rouge-l']['f'])
    st = 0
    sum_r_1, sum_r_2, sum_r_l = 0, 0, 0
    for count_len in count_tgts:
        sum_r_1+=(sum(rouge1f_score[st:st+count_len]))/count_len
        sum_r_2+=(sum(rouge2f_score[st:st+count_len]))/count_len
        sum_r_l+=(sum(rougelf_score[st:st+count_len]))/count_len
        st+=count_len
    r_1_s, r_2_s, r_l_s = round(sum_r_1/len(count_tgts)*100, 2), round(sum_r_2/len(count_tgts)*100, 2), round(sum_r_l/len(count_tgts)*100, 2)

    bleu_1 = corpus_bleu(peers, models, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(peers, models, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(peers, models, weights=(0.333, 0.333, 0.334, 0))
    c_bleu = corpus_bleu(peers, models)
    bs = len(peers)
    s_bleu, meteor = 0, 0
    for i in range(bs):
        s_bleu += sentence_bleu(peers[i], models[i])
        meteor += meteor_score(peers[i], models[i])
    s_bleu = round(s_bleu/bs*100, 2)
    meteor = round(meteor/bs*100, 2)

    p, r, f1 = score(xxsrcs, xxtgts, lang='zh', verbose=True, device=device) # max
    st, mean_f1 = 0, 0
    for count_len in count_tgts:
        mean_f1 += (sum(f1[st:st+count_len]))/count_len
        st += count_len
    mean_f1 = mean_f1/len(count_tgts)*100
    
    print_log("ROUGE-1: ", r_1_s, "ROUGE-2: ", r_2_s, "ROUGE-l: ", r_l_s)
    print_log("BLEU-1: ", round(bleu_1*100, 2), "BLEU-2: ", round(bleu_2*100, 2), "BLEU-3: ", round(bleu_3*100, 2))
    print_log("S-BLEU:", s_bleu, "C-BLEU:", round(c_bleu*100, 2))
    print_log("METEOR:", meteor)
    print_log("BERTScore:", mean_f1)

    return r_2_s

def srouge_eval(peers, models, print_log=print):
    assert len(peers) == len(models)
    xsrcs, xtgts = [], []
    count_tgts = []
    for s, ts in zip(models, peers):
        count_tgts.append(len(ts))
        for t in ts:
            xsrcs.append(s)
            xtgts.append(t)
    xxsrcs, xxtgts = [], []
    for s, t in zip(xsrcs, xtgts):
        xxsrcs.append(" ".join(list(s)))
        xxtgts.append(" ".join(list(t)))

    scores = Rouge(metrics=["rouge-l"]).get_scores(xxsrcs, xxtgts, avg=False)
    rouge1f_score, rouge2f_score, rougelf_score = [], [], []
    for indx, one in enumerate(scores):
        rouge1f_score.append(rouge_n(xxsrcs[indx].split(), xxtgts[indx].split(), 1))
        rouge2f_score.append(rouge_n(xxsrcs[indx].split(), xxtgts[indx].split(), 2))
        rougelf_score.append(one['rouge-l']['f'])
    st = 0
    sum_r_1, sum_r_2, sum_r_l = 0, 0, 0
    for count_len in count_tgts:
        sum_r_1+=(sum(rouge1f_score[st:st+count_len]))/count_len
        sum_r_2+=(sum(rouge2f_score[st:st+count_len]))/count_len
        sum_r_l+=(sum(rougelf_score[st:st+count_len]))/count_len
        st+=count_len
    r_1_s, r_2_s, r_l_s = round(sum_r_1/len(count_tgts)*100, 2), round(sum_r_2/len(count_tgts)*100, 2), round(sum_r_l/len(count_tgts)*100, 2)

    bleu_1 = corpus_bleu(peers, models, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(peers, models, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(peers, models, weights=(0.333, 0.333, 0.334, 0))
    c_bleu = corpus_bleu(peers, models)
    
    print_log("ROUGE-1: ", r_1_s, "ROUGE-2: ", r_2_s, "ROUGE-l: ", r_l_s)
    print_log("BLEU-1: ", bleu_1, "BLEU-2: ", bleu_2, "BLEU-3: ", bleu_3, "BLEU-4:", c_bleu)

    return r_2_s

def _srouge(peers, models, print_log=print):
    assert len(peers) == len(models)
    xsrcs, xtgts = [], []
    count_tgts = []
    for s, ts in zip(models, peers):
        count_tgts.append(len(ts))
        for t in ts:
            xsrcs.append(s)
            xtgts.append(t)
    xxsrcs, xxtgts = [], []
    for s, t in zip(xsrcs, xtgts):
        xxsrcs.append(" ".join(list(s)))
        xxtgts.append(" ".join(list(t)))

    scores = Rouge().get_scores(xxsrcs, xxtgts, avg=True)
    f_score = [round(scores["rouge-1"]['f'] * 100, 2),
                round(scores["rouge-2"]['f'] * 100, 2),
                round(scores["rouge-l"]['f'] * 100, 2)]
    # print
    print_log("ROUGE-1: ", f_score[0], "ROUGE-2: ", f_score[1], "ROUGE-l: ", f_score[2])
    return f_score[1]

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
def get_all_score(file_1, file_2):
    word_dict = {}
    idx = 0
    file_1_word, file_2_word = [], []
    with open(file_1, "r") as f:
        for s in f:
            for word in s.strip().split():
                if word not in word_dict:
                    word_dict[word] = idx
                    idx += 1
    with open(file_2, "r") as f:
        for s in f:
            for word in s.strip().split():
                if word not in word_dict:
                    word_dict[word] = idx
                    idx += 1
                    
    with open(file_1, "r") as f:
        for s in f:
            one_hot = np.zeros(len(word_dict))
            for word in s.strip().split():
                one_hot[word_dict[word]] = 1
            file_1_word.append(one_hot)
    with open(file_2, "r") as f:
        for s in f:
            one_hot = np.zeros(len(word_dict))
            for word in s.strip().split():
                one_hot[word_dict[word]] = 1
            file_2_word.append(one_hot) 
    p = precision_score(file_2_word , file_1_word,average='macro')
    r = recall_score(file_2_word, file_1_word, average='macro')
    f1 = f1_score(file_2_word, file_1_word, average='macro')
    accuracy = accuracy_score(file_1_word,file_2_word)
    print("precision: {:.4f}, recall: {:.4f}, f1_score: {:.4f}, accuracy: {:.4f}".format(p*100, r*100, f1*100, accuracy*100))

