# Copyright (c) 2018 British Broadcasting Corporation

from Crypto.Cipher import AES
from Crypto import Random

BLOCK_SIZE = 1024 * AES.block_size


def open_file(filename, password):
    """
    Get a generator which provides lines of data from a file.
    - if a password is provided the file is decrypted.
    
    The encryption is assumed to be AES-256 using:
        openssl enc -e -aes-256-cbc -salt -in <infile> -out <outfile>
    
    """
    stream = get_decrypted_stream(filename, password)
    line_generator = get_line_generator(stream)    
    return line_generator

    
def get_line_generator(stream):
    """
    Read lines of data from a stream of bytes objects
    
    """
    # The residual is used to retain data at the end of a chunk that doesn't end with a newline
    residual = b''
    
    for chunk in stream:   
        # Split the chunk into lines
        lines = chunk.split(b'\n')
        
        for i in range(0, len(lines)):
            line = lines[i]
            
            # The first element of split never starts with a newline so add residual from the previous chunk
            if i == 0:
                line = residual + line
                residual = b''
            
            # The last element of split never ends with a newline
            if i == len(lines) - 1:
                residual = residual + line
            
            else:
                yield line + b'\n'
    
    if len(residual) > 0:
        yield residual


def get_decrypted_stream(encryptedFilename, password):
    """
    Read data from an encrypted file.
    
    """
    salt = get_salt(encryptedFilename)
    
    # Now create the key and iv.
    key, iv = get_key_and_iv(password, salt)
    
    if key is None:
        return None

    cipher = AES.new(key, AES.MODE_CBC, iv)
    started = False
    finished = False
    next_chunk = ''
    
    with open(encryptedFilename, "rb") as infile:
        while not finished:
            encryptedChunk = infile.read(BLOCK_SIZE)
            
            if started == False:
                encryptedChunk = encryptedChunk[16:] 
            
            chunk, next_chunk = next_chunk, cipher.decrypt(encryptedChunk)
            
            if len(next_chunk) == 0:
                if isinstance(chunk, str):
                    padding_len = ord(chunk[-1])
                else:
                    padding_len = chunk[-1]
            
                chunk = chunk[:-padding_len]
                finished = True
            
            if started:
                yield chunk
            
            started = True
    
    
def get_salt(filename):
    with open(filename, 'rb') as infile:
        raw = infile.read(16)
        assert(raw[:8] == b'Salted__')
        salt = raw[8:16]        
        return salt
    
    
def get_key_and_iv(password, salt, klen=32, ilen=16):
    '''
    Derive the key and the IV from the given password and salt.
    From: http://stackoverflow.com/questions/13907841/implement-openssl-aes-encryption-in-python

    @param password  The password to use as the seed.
    @param salt      The salt.
    @param klen      The key length.
    @param ilen      The initialization vector length.
    '''
    msgdgst='md5'
    mdf = getattr(__import__('hashlib', fromlist=[msgdgst]), msgdgst)
    password = password.encode('ascii', 'ignore')  # convert to ASCII

    try:
        maxlen = klen + ilen
        keyiv = mdf(password + salt).digest()
        tmp = [keyiv]
        
        while len(tmp) < maxlen:
            tmp.append( mdf(tmp[-1] + password + salt).digest() )
            keyiv += tmp[-1]  # append the last byte
        
        key = keyiv[:klen]
        iv = keyiv[klen:klen+ilen]
        return key, iv
    
    except UnicodeDecodeError:
        return None, None
