""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols import symbols

# I want to change this.

# Mappings from symbol to numeric ID and vice versa:
## IPA
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

## uroman
_uroman_to_id = {s: i for i, s in enumerate(romans)}
_id_to_uroman = {i: s for i, s in enumerate(romans)}


def uromanize(text, uroman_pl):
  iso = "xxx"
  with tempfile.NamedTemporaryFile() as tf, \
        tempfile.NamedTemporaryFile() as tf2:
      with open(tf.name, "w") as f:
          f.write("\n".join([text]))
      cmd = f"perl " + uroman_pl
      cmd += f" -l {iso} "
      cmd +=  f" < {tf.name} > {tf2.name}"
      os.system(cmd)
      outtexts = []
      with open(tf2.name) as f:
          for line in f:
              line =  re.sub(r"\s+", " ", line).strip()
              outtexts.append(line)
      outtext = outtexts[0]
  return outtext


# text_to_sequence를 -> 1. text_to_phoneme 2. phoneme_to_sequence로 나눠주기
def text_to_phoneme(text, type_of_phoneme, lang):
  ''' Coverts a string of texts to symbols(IPA, uroman,,)
    Args:
      text: string to convert to symbols
      type_of_phoneme: type of text representation to replace texts
    Returs:
      string of phonemes
  '''
  if type_of_phoneme == "IPA":
    IPA = phonemize(text, language=lang, backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
    # English TTS model만 사용할 것이므로 english가 아닌 text는 English에서 사용되지 않는 phoneme을 similar한 것으로 mapping해주고자 함.
    if lang != "en":
      _IPA = [i for i in IPA]
      _IPA = " ".join(_IPA)
      new_IPA = similar_ipa(_IPA, lang)
      new_IPA = new_IPA.replace("   ", "_")
      new_IPA = new_IPA.replace(" ", "")
      phonemes = new_IPA.replace("_", " ")

  elif type_of_phoneme == "uroman":
    uroman_dir = "/home/solee0022/tts-asr/uroman/"
    uroman_pl = os.path.join(uroman_dir, "bin", "uroman.pl")
    print(f"uromanize")
    phonemes = uromanize(text, uroman_pl)

  return phonemes


def phoneme_to_sequence(phonemes, type_of_phoneme):
  '''Converts a string of phoneme(IPA, uroman,,) to a sequence of IDs corresponding to the symbols in the text.
    Args:
      phonemes: string to convert to a sequence
    Returns:
      List of integers corresponding to the phoneme in the text
  '''
  sequence = []

  if type_of_phoneme == "IPA":
    for symbol in phonemes:
      symbol_id = _symbol_to_id[symbol]
      sequence += [symbol_id]
  
  elif type_of_phoneme == "uroman":
    for roman in phonemes:
      roman_id = _uroman_to_id[roman]
      sequence += [roman_id]

  return sequence


def cleaned_text_to_sequence(cleaned_text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text
