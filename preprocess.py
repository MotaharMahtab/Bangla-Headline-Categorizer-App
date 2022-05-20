from normalizer import normalize
import regex as re

def clean_title(title):
  title =str(title)
  while re.search('[\u0980-\u09ff][\'\’\‘\”\“\,\\.]+[\u0980-\u09ff]', title):
      pos = re.search('[\u0980-\u09ff][\'\’\‘\”\“\,\\.]+[\u0980-\u09ff]', title).start()
      title = title[:pos+1] + title[pos+2:]
  title = re.sub(r"[\’\‘\”\“]+", "'", title)
  title = re.sub(r"[\*\#\;]+", "", title)
  title = re.sub(r'আরো পড়ুন.*','',title,flags=re.U|re.S) 
  title = normalize(title,url_replacement='',emoji_replacement='') 
  return title