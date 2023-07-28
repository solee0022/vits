from collections import defaultdict, OrderedDict
import panphon.distance

from text.symbols import symbols

dst = panphon.distance.Distance()


# korean = ["a", "b", "c", "d", "u"]
# english = ["a", "b:", "c:", "dz"]

# symbols_dict
lang_symbols = {}
lang_symbols["en"] = symbols
lang_symbols["ko"]= "ɡɫ\"sʃwlʌjkbʑutɕpŋhnoɯdˈɾm-ˌqe, .ɐɛ?i"
# lang_symbols["pt"] = ""


def make_mapping_table(lang):
    lang_list = []
    _lang_symbols = lang_symbols[lang]
    lang_mapping_table = {}

    english = [i for i in lang_symbols["en"]]

    for i in _lang_symbols:
        if i not in english:
            lang_list.append(i)
        else:
            lang_mapping_table[i] = i


    for _lang in lang_list:
    
        minimum_dist = 99
        closest_toksen = None
    
        for en in english:
            dist = dst.hamming_feature_edit_distance(_lang, en)
            minimum_dist = min(minimum_dist, dist)

            if minimum_dist == dist: # Hit
                print(dist, en, _lang)
                closest_token = en
    
        lang_mapping_table[_lang] = closest_token

    # blank space
    lang_mapping_table[""] = ""

    return lang_mapping_table



# Example -> function

def similar_ipa(sentence, lang):
    #sentence = "a b c d a b c d u"

    # mapping table for specific languages
    lang_mapping_table = make_mapping_table(lang)

    new_sentence = ""
    for token in sentence.split(" "):
        new_token = lang_mapping_table[token]

        new_sentence += " " + new_token

    return new_sentence

