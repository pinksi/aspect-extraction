import os
import sys
import spacy
import pandas as pd

from tqdm import tqdm
import math
from collections import Counter

def apply_extraction(review_body, nlp):
    doc = nlp(review_body)
    
    # rule1
    r1_pairs = []
    for token in doc:
        if token.dep_ in ("amod", "advmod") and token.head.pos_ == "NOUN":
            r1_pairs.append((token.head.text, token.text))
            
    # rule2 - Ex: The phone is very lightweight to carry.
    r2_pairs = []
    for token in doc:
        A = ""
        M = ""
        for child in token.children:
            if child.dep_ == "nsubj" and child.pos_ == "NOUN":
                A = child.text
                
            if not M and child.dep_ == "dobj":
                M = child.text
            
            if not M and child.dep_ == "acomp":
                M = child.text
        if A and M:
            r2_pairs.append((A, M))
            
    # rule3
    r3_pairs = []
    prev_token_pos = ""
    for token in doc:
        # print(token, [i for i in token.children])
        A = ""
        M = ""
        for child in token.children:
            if token.pos_ == "NOUN" and child.dep_ == "prep":
                A = token.text
            # if prev_token_pos != "NOUN" and token.dep_ =="prep" and token.pos_ == "ADP" and child.dep_ == "pobj":
            #     A = child.text
        if A and not M:
            r3_pairs.append((A, M))
        prev_token_pos = token.pos_

            
    # rule4
    r4_pairs = []
    for token in doc:
        children = token.children
        A = ""
        M = ""
        for child in children:
            if child.dep_ == "nsubjpass":
                A = child.text
                
            if child.dep_ == "advmod":
                M = child.text
        
        if A and M:
            r4_pairs.append((A, M))
            
    # rule5
    r5_pairs = []
    for token in doc:
        children = token.children
        A = ""
        M = ""
        for child in children:
            if child.dep_ == "nsubj":
                A = child.text
                
            if child.dep_ == "cop":
                M = child.text
        
        if A and M:
            r5_pairs.append((A, token.text))


    # rule 6 - Ex. I like the lens of the screen.
    r6_pairs = []
    for token in doc:
        A = ""
        M = ""
        if token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ == "nsubj" and child.pos_ == "PRON":
                    continue
                if child.dep_ == "dobj" and child.pos_ == "NOUN":
                    A = child.text
                    M = token.text
        if A and M:
            r6_pairs.append((A, M))
                
                
    # rule 7 - I would like to comment on the camera of this phone.
    A = ""
    M= ""
    r7_pairs=[]
    for i in range(len(doc)-1):
        if doc[i].pos_ == "VERB" and doc[i+1].pos_ == "ADP" and doc[i+1].dep_ == "prep":
            for token in doc[i+1:]:
                for child in token.children:
                    if child.dep_ == "pobj" and child.pos_ == "NOUN":
                        A = child.text
                        M = doc[i].text
                break   
        if A and M:
            r7_pairs.append((A, M))
            break
            
    # rule 8 - It is easy to use
    r8_pairs = []
    for token in doc:
        A = ""
        M = ""
        if token.pos_ == "AUX" and len([child for child in token.children]) >= 2:
            for child in token.children:
                if child.dep_ == "acomp" and child.pos_ == "ADJ":
                    M = child.text
                if child.dep_ == "xcomp" and child.pos_ in ("VERB", "NOUN"):
                    A = child.text
            if A and M:
                r8_pairs.append((A, M))
            
    # aspects = []
    aspects_pairs = r1_pairs + r2_pairs + r3_pairs + r4_pairs + r5_pairs
    # aspect_dict = {"review_id": review_id, "review_body": review_body, "aspect_pairs": aspects}
    return aspects_pairs

def extract_aspects(pairs):
    """
    extract aspects from (aspects, opinion words) tuple
    """
    aspects = list(set([i[0] for i in pairs]))
    filtered_aspects = []
    for w in aspects:
        # remove pronouns from aspects
        if nlp(w)[0].pos_ != "PRON":
            filtered_aspects.append(w)
    return filtered_aspects

def extract_noun_chunk(review, nlp):
    """
    get noun chunk for review
    """
    nchunks = []
    for np in nlp(review).noun_chunks:
        nchunks.append(" ".join([token.text for token in np if not token.is_stop]))
    return nchunks

def find_noun_phrase(aspects, noun_chunks):
    """
    find corresponding noun phrase for extracted aspect (noun)
    """
    # get noun phrase for the aspects 
    # new_aspects = []
    # for ap in aspects:
    #     new = ""
    #     for nc in noun_chunks:
    #         if ap not in nc.split(" "):
    #             continue
    #         else:
    #             new = nc
    #             break
    #     if new:
    #         new_aspects.append(new)
    #     else:
    #         new_aspects.append(ap)
        
                
    # for small sentences, usually the noun phrases are aspects
    if not aspects and len(noun_chunks) <= 2:
        return [i for i in noun_chunks if i]
    # return new_aspects
    return aspects

def find_similarity(l1, l2):
    c1 = Counter(l1)
    c2 = Counter(l2)
    
    # cosine similarity
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)

def compute_precision(reviews):
    TP = 0
    FP = 0 
    FN = 0
    TN = 0
    for data in reviews:
        # extracted
        for word in data["extracted_aspects"]:
            # and are actual aspects
            if word in data["actual_aspects"]:
                TP += 1
            # not actual aspects
            if word not in data["actual_aspects"]:
                FP += 1
        # not extracted - but are aspects
        for word in data["actual_aspects"]:
            if word not in data["extracted_aspects"]:
                FN += 1
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

def main():
    file_path = "datasets/Restaurant_reviews/Restaurants_Train_v2.csv"
    raw_data = pd.read_csv(file_path)
    # print(raw_data.head())
    reviews = raw_data[["id", "Sentence", "Aspect Term"]]
    # reviews.head()
    reviews.rename(columns={"id": "id", "Sentence": "text", "Aspect Term": "original_aspects"}, inplace=True)
    # list all the aspects of a sentence in one column
    results = {}
    for row in reviews.itertuples():
        if row.text in results:
            results[row.text].append(row.original_aspects)
        else:
            results[row.text] = [row.original_aspects]

    # proper formatting
    all_reviews = []
    for key, val in results.items():
        all_reviews.append({"review": key, "actual_aspects": list(set(val))}) # getting unique actual aspects

    nlp = spacy.load("en_core_web_lg")
    for data in tqdm(all_reviews):
        extracted_aspects = extract_aspects(apply_extraction(data["review"], nlp))
        noun_chunks = extract_noun_chunk(data["review"], nlp)
        data["extracted_aspects"] = find_noun_phrase(extracted_aspects, noun_chunks)
        try:
            data["similarity"] = find_similarity(data["actual_aspects"], data["extracted_aspects"])
        except ZeroDivisionError:
            data["similarity"] = 0
        if data["similarity"] >=0.8:
            data["check"] = True
        else:
            data["check"] = False

    precision, recall, f1 = compute_precision(all_reviews)
    print("Precision: ", precision, "Recall: ", recall, "F1-score: ", f1)

    pd.DataFrame(all_reviews).to_csv("output/extracted_aspects_v3.csv")

if __name__ == "__main__":
    main()