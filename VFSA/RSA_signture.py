from Crypto.PublicKey import RSA
import Crypto.Signature.PKCS1_v1_5 as sign_PKCS1_v1_5 
from Crypto import Hash

def to_verify_with_public_key(signature, plain_text):
    '''
    Public key check algorithm
    '''
    with open("d.pem", "rb") as x:
        my_public_key = x.read() 
    verifier = sign_PKCS1_v1_5.new(RSA.importKey(my_public_key))
    _rand_hash = Hash.SHA256.new()
    _rand_hash.update(plain_text.encode())
    verify = verifier.verify(_rand_hash, signature)
    return verify 

def to_sign(plain_text):
    '''
    Private key signature algorithm
    '''
    with open("c.pem", "rb") as x:
        my_private_key = x.read() 
    signer_pri_obj = sign_PKCS1_v1_5.new(RSA.importKey(my_private_key))
    rand_hash = Hash.SHA256.new()
    rand_hash.update(plain_text.encode())
    signature = signer_pri_obj.sign(rand_hash)
    return signature 
