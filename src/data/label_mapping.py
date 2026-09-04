sentiment_mapping = {
    "positive": 0,
    "negative": 1,
    "neutral": 2,
    "none": 2
}

hate_mapping = {
    # Non-offensive / Neutral class: 0
    "not_offensive": 0,
    "non-antireligion": 0,
    "not": 0,
    "not-kannada": 0,
    "not-malayalam": 0,
    "not-tamil": 0,
    "none": 0,
    
    # Offensive / Hate class: 1
    "offensive_targeted_insult_group": 1,
    "offensive_targeted_insult_individual": 1,
    "offensive_targeted_insult_other": 1,
    "offensive_untargetede": 1,
    "antireligion": 1,
    "off": 1
}

def map_sentiment(label):
    if not isinstance(label, str):
        return None
    l = label.strip().lower()
    return sentiment_mapping.get(l, None)

def map_hate(label):
    if not isinstance(label, str):
        return None
    l = label.strip().lower()
    return hate_mapping.get(l, None)

