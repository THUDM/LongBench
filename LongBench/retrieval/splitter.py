import re
def split_long_sentence(sentence, regex, chunk_size=200, filename='Unknown'):
    chunks = []
    sentences = re.split(regex, sentence)
    current_chunk = ""
    for s in sentences:
        if current_chunk and get_word_len(current_chunk) + get_word_len(s) <= chunk_size:
            current_chunk += ' ' if s == '' else s
        else:
            if current_chunk:
                chunks.append(current_chunk)
                # if (len(current_chunk) > chunk_size*5):
                current_len = get_word_len(current_chunk)
                if (current_len > chunk_size * 1.5):
                    print(f"\n{filename}-{len(chunks)-1} Chunk size: {current_len}")
                
            current_chunk = s
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def get_word_list(s1):
    # Separate sentences by word, Chinese by word, English by word, numbers by space
    regEx = re.compile('[\W]')   
    res = re.compile(r"([\u4e00-\u9fa5])")    #  [\u4e00-\u9fa5] for Chinese

    p1 = regEx.split(s1.lower())
    str1_list = []
    for str in p1:
        if res.split(str) == None:
            str1_list.append(str)
        else:
            ret = res.split(str)
            for ch in ret:
                str1_list.append(ch)

    list_word1 = [w for w in str1_list if len(w.strip()) > 0]  

    return  list_word1
def get_word_len(s1):
    return len(get_word_list(s1))

regex = r'([。？！；\n.!?;]\s*)'
