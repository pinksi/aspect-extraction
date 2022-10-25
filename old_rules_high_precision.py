def apply_extraction(review_body, nlp):
    # review_body = row["Sentence"]
    # review_id = row["id"]
    
    doc = nlp(review_body)
            
    # rule1 - Ex: The phone is very lightweight to carry.
    r1_pairs = []
    for token in doc:
        A = ""
        M = ""
        for child in token.children:
            if child.dep_ == "nsubj" and child.pos_ == "NOUN":
                A = child.text
                
                if not M and child.dep_ == "dobj" and child.pos_ != "NOUN":
                    M = child.text

                if not M and child.dep_ == "acomp":
                    M = child.text
        if A and M:
            r1_pairs.append((A, M))
            
    # rule1
    r2_pairs = []
    for token in doc:
        if token.dep_ in ("amod", "advmod") and token.head.pos_ == "NOUN":
            A = token.head.text
            # check if it is a noun phrase or not
            for c in token.head.children:
                if c.pos_ == "NOUN" and c.dep_ == "compound":
                    A = c.text + " " + A
                    M = token.text
                if A and M:
                    r2_pairs.append((A, M))
            
    # rule3
    r3_pairs = []
    # prev_token_pos = ""
    for token in doc:
        A = ""
        M = ""
        for child in token.children:
            if token.pos_ == "NOUN" and child.dep_ == "prep":
                A = token.text
                M = child.text
            # if prev_token_pos != "NOUN" and token.dep_ =="prep" and token.pos_ == "ADP" and child.dep_ == "pobj":
            #     A = child.text
        if A and not M:
            r3_pairs.append((A, M))
        # prev_token_pos = token.pos_

            
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
                    phrase1 = child.text
                    # check if it is a noun phrase or not
                    for c in child.children:
                        if c.pos_ == "NOUN" and c.dep_ == "compound":
                            A = c.text + " " + phrase1
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
    # print(r1_pairs, r2_pairs, r3_pairs, r4_pairs, r5_pairs, r6_pairs, r7_pairs, r8_pairs)
    aspects_pairs = r1_pairs + r2_pairs + r3_pairs + r4_pairs + r5_pairs + r6_pairs + r7_pairs + r8_pairs
    # aspect_dict = {"review_id": review_id, "review_body": review_body, "aspect_pairs": aspects}
    return list(set(aspects_pairs))