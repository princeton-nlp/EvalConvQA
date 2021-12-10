from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from ncr.replace_corefs import resolve
from collections import Counter
import string
import difflib

from numpy import True_

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")
PRONOUNS = {
    'all', 'another', 'any', 'anybody', 'anyone', 'anything', 'as', 'aught', 'both', 'each other', 'each', 'either',
    'enough', 'everybody', 'everyone', 'everything', 'few', 'he', 'her', 'hers', 'herself', 'him', 'himself', 'his',
    'i', 'idem', 'it', 'its', 'itself', 'many', 'me', 'mine', 'most', 'my', 'myself', 'naught', 'neither', 'no one',
    'nobody', 'none', 'nothing', 'nought', 'one another', 'one', 'other', 'others', 'ought', 'our', 'ours', 'ourself',
    'ourselves', 'several', 'she', 'some', 'somebody', 'someone', 'something', 'somewhat', 'such', 'suchlike', 'that',
    'thee', 'their', 'theirs', 'theirself', 'theirselves', 'them', 'themself', 'themselves', 'there', 'these', 'they',
    'thine', 'this', 'those', 'thou', 'thy', 'thyself', 'us', 'we', 'what', 'whatever', 'whatnot', 'whatsoever',
    'whence', 'where', 'whereby', 'wherefrom', 'wherein', 'whereinto', 'whereof', 'whereon', 'wheresoever', 'whereto',
    'whereunto', 'wherever', 'wherewith', 'wherewithal', 'whether', 'which', 'whichever', 'whichsoever', 'who',
    'whoever', 'whom', 'whomever', 'whomso', 'whomsoever', 'whose', 'whosesoever', 'whosever', 'whoso', 'whosoever',
    'ye', 'yon', 'yonder', 'you', 'your', 'yours', 'yourself', 'yourselves'
}
ARTICLES = {'the','this','that','these','those','a','an'}

def get_overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2))
    return s1[pos_a:pos_a+size], pos_a+size,pos_b+size

def resolve_coreference(text = ""):

    result_dict = predictor.predict(document=text)

    text_formatted= {
    'document': result_dict['document'],
    'clusters': result_dict['clusters']
    }

    resolved_toks = resolve(text_formatted['document'], text_formatted['clusters'])
    replaced_text = ' '.join(resolved_toks)
    return replaced_text

def is_entity(word):

    tokens = word.split()
    tokens = [t for t in tokens if (t not in ARTICLES)]
    is_entity = True if all([t[0].isupper() for t in tokens]) else False
    return is_entity

def find_coreference_f1s(text1="", text2="", skip_entity=True):

    result_dict1 = predictor.predict(document=text1)
    result_dict2 = predictor.predict(document=text2)
    text_formatted1 = {
        'document': result_dict1['document'],
        'clusters': result_dict1['clusters']
    }
    text_formatted2 = {
        'document': result_dict2['document'],
        'clusters': result_dict2['clusters']
    }

    print("Cluster1:",text_formatted1['clusters'])
    print("Cluster2:",text_formatted2['clusters'])
    q_start1 = max(idx for idx, val in enumerate(
        text_formatted1['document']) if val == '>')+1
    subs1 = []
    for cluster in text_formatted1['clusters']:
        for r in cluster:
            if r[0] >= q_start1:
                subs1.append(cluster)
                break
    subs1.sort(key=lambda c: c[-1][0])
    print("Subs1:",subs1)

    q_start2 = max(idx for idx, val in enumerate(
        text_formatted2['document']) if val == '>')+1
    subs2 = []
    for cluster in text_formatted2['clusters']:
        for r in cluster:
            if r[0] >= q_start2:
                subs2.append(cluster)
                break

    subs2.sort(key=lambda c: c[-1][0])
    print("Subs2:",subs2)

    nouns1 = []
    for cluster in subs1:
        cluster_strings = list(map(lambda x: " ".join(
            text_formatted1['document'][x[0]:x[1]+1]), cluster))
        if skip_entity and is_entity(cluster_strings[-1]):
            continue
        
        set_strings = set([s for s in cluster_strings if s.lower() not in PRONOUNS])

        span_lens = list(map(len, set_strings))
        head_span_idx = None
        for i, span_len in enumerate(span_lens):
            if span_len > 0:
                head_span_idx = i
                break
        if head_span_idx is None:
            nouns1.append(cluster_strings[0])
        else:
            nouns1.append(list(set_strings)[head_span_idx])

    nouns2 = []
    for cluster in subs2:
        cluster_strings = list(map(lambda x: " ".join(
            text_formatted2['document'][x[0]:x[1]+1]), cluster))
        if skip_entity and is_entity(cluster_strings[-1]):
            continue

        set_strings = set([s for s in cluster_strings if s.lower() not in PRONOUNS])

        span_lens = list(map(len, set_strings))
        head_span_idx = None
        for i, span_len in enumerate(span_lens):
            if span_len > 0:
                head_span_idx = i
                break
        if head_span_idx is None:
            nouns2.append(cluster_strings[0])
        else:
            nouns2.append(list(set_strings)[head_span_idx])

    resolved_toks1 = resolve(
        text_formatted1['document'], text_formatted1['clusters'])

    resolved_toks2 = resolve(
        text_formatted2['document'], text_formatted2['clusters'])

    f1s = []

    def f1(lst1, lst2):
        common = Counter(lst1) & Counter(lst2)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(lst2)
            recall = 1.0 * num_same / len(lst1)
            f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if len(nouns1) == 0 or len(nouns2) == 0 or len(nouns1) != len(nouns2):
        f1s = [0] * max(len(nouns1), len(nouns2))
    else:
        short_list, long_list = nouns1, nouns2
        for i in range(len(long_list)):
            max_f1 = 0
            for j in range(len(short_list)):
                f1_temp = f1(long_list[i].lower().split(), short_list[j].lower().split())
                if f1_temp > max_f1:
                    max_f1 = f1_temp
            f1s.append(max_f1)

    return f1s, " ".join(resolved_toks1), " ".join(resolved_toks2)

if __name__ == "__main__":
    s1 = "Why did he fight the Dutch? Dutch colonial rule was becoming unpopular among local farmers because of tax rises, crop failures Is there any interesting information? Diponogoro was widely believed to be the Ratu Adil, the just ruler predicted in the Pralembang Jayabaya. <Q> What is the Pralembrang Jayabaya?"
    s2 = "Why did he fight the Dutch? due to their lack of coherent strategy and commitment in fighting Diponegoro's guerrilla warfare. Is there any interesting information? CANNOTANSWER <Q> What is the Pralembrang Jayabaya?"
    s3 = "How many shows did she do after her comeback? On 12 July 2012, Reddy returned to the musical stage at Croce's Jazz Bar in San Diego and for a benefit concert for the arts at St. Genevieve High School Did she perform anywhere after that? Reddy appeared in downtown Los Angeles at the 2017 Women's March on January 21. <Q> What did she sing at the Womens March?"
    s4 = "How many shows did she do after her comeback? CANNOTANSWER Did she perform anywhere after that? Reddy performed at the Paramount nightclub at The Crown & Anchor in Provincetown on 13 October 2013. <Q> What did she sing at the Womens March?"
    print(find_coreference_f1s(s3,s4))

    
