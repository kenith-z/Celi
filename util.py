def encode_word(bd, word):
    """
    字转整数
    """
    encoding = bd.get_token(word)
    if encoding  is not None:
        return [encoding]
    else:
        
        return [bd.get_token("*")]
    
def decode_word(bd, token):
    """
    整数转字
    """
    encoding = bd.get_word(token)
    if encoding is not None:
        return encoding
    else:
        return "*"
    
