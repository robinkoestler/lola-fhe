
import time
import numpy as np
load("sagefhepoly/polyfhe.sage")
load("FHE_DFT_class.sage")

class LOLA:
    @classmethod
    def setup(cls, N, slots, modulus, precision=53, kappa=2, h=2):
        cls.N = N
        assert slots & slots-1 == 0 and N//2 >= slots >= 1, "Wrong number of slots"
        cls.n, cls.log_n = slots, log(slots, 2)
        cls.prec = precision
        cls.delta = 2 ** precision
        cls.kappa = kappa
        cls.q = modulus
        assert h < cls.N, "h must be smaller than N"
        assert h & h-1 == 0, "h must be a power of 2"
        assert cls.N // (cls.n * h * 2) >= 1, "N/(2*slots*hw) must be >=1"
        cls.h, cls.log_h = h, ZZ(log(h, 2))
        cls.max_levels = 1 + log(h, 2) + cls.log_n
        cls.q_L = cls.q * (2**(precision * cls.max_levels)) # largest regular modulus (full level)
        cls.Pq_L = cls.q_L * cls.q_L # P = q_L, approximately, as in CKKS
        cls.Encoder = FHE_DFT(N, slots, modulus, precision) # see file FHE_DFT_class.sage
        cls.gen_secret(h=h)
        cls.gen_switching_keys()
        
    @classmethod
    def gen_secret(cls, h, block=True):
        if block:
            cls.s = np.append([1], np.zeros(cls.N - 1))
            cls.blocks = cls.N // h
            cls.f = cls.blocks // 2 // cls.n
            for i in range(1, h):
                cls.s[i * (cls.blocks) + np.random.randint(cls.blocks)] = 1
        else:
            cls.s = np.concatenate((np.ones(h), np.zeros(cls.N - h)))
            np.random.shuffle(cls.s)
        # Below we precompute the secret key in blocks, necessary for bootstrapping
        cls.s_in_blocks = [cls.block_slice(cls.s, i) for i in range(2*cls.n)]
        cls.s_in_blocks = [cls.Encoder.encode_clear(i, modulus=0) for i in cls.s_in_blocks]
        cls.s = Poly(cls.s, 0)
        cls.s_block_enc = [LOLA.encrypt(i, cls.q_L) for i in cls.s_in_blocks]
        cls.boot_delta = ZZ(round(cls.delta * ((cls.q * cls.n / (pi * 4 * cls.delta)) ** (1 / cls.h))))
        
    @classmethod
    def gen_switching_keys(cls):
        assert hasattr(cls, "s"), "secret key not generated"
        # the evaluation key for ciphertext multiplication, encrypting s^2
        evk = Poly(((cls.s % cls.Pq_L) ** 2) * cls.q_L, cls.q_L)
        cls.evk = LOLA.encrypt(evk, cls.Pq_L)
        
        # we generate the keyswitching keys for the automorphisms of powers of 2, with pos. and neg. sign
        cls.ksk = {}
        for index in [1 << i for i in range(cls.N.bit_length()-2)]:
            for j in range(-1, 2, 2): # j = -1, 1
                newkey = Poly((cls.s.auto(index * j) * cls.q_L) % cls.Pq_L, cls.Pq_L)
                cls.ksk[str(index * j)] = LOLA.encrypt(newkey, cls.Pq_L)
        # ...and for the conjugation automorphism        
        newkey =  Poly((cls.s.auto_inverse() * cls.q_L) % cls.Pq_L, cls.Pq_L)
        cls.ksk_conj = LOLA.encrypt(newkey, cls.Pq_L)
        
    ## INIT    
        
    def __init__(self, args) -> None:
        self.a, self.b = args
    
    # ARITHMETIC OPERATIONS

    def __add__(self, other):
        if isinstance(other, Poly): return LOLA([self.a, self.b + other])
        return LOLA([self.a + other.a, self.b + other.b])
     
    def __neg__(self): return LOLA([-self.a, -self.b])

    def __sub__(self, other):
        if isinstance(other, Poly): return LOLA([self.a, self.b - other])
        return LOLA([self.a - other.a, self.b - other.b])


    def __mul__(self, other):
        if isinstance(other, LOLA):
            assert self.a.modulus == other.a.modulus, f"Moduli must be the same (and = {self.q})"    
            d0 = (self.a * other.a) % self.Pq_L
            d1 = self.a * other.b + self.b * other.a
            d2 = self.b * other.b
            d0 = (self.evk * d0).scale(self.q_L) % d1.modulus
            return (LOLA([d1, d2]) + d0)
        assert not isinstance(other, int), "int multiplication not implemented"
        
        m1, m2 = self.modulus, other.modulus # Below, we figure the right modulus to continue
        if m1 == 0 and m2 != 0: self = self % m2
        elif m2 == 0 and m1 != 0: other = other % m1
        elif m2 != 0 and m1 != 0:
            modulus = min(m1, m2)
            self, other = self % modulus, other % modulus            
        return LOLA([self.a * other, self.b * other]) 
    
    def __radd__(self, other): return self + other
    def __rsub__(self, other): return self - other
    def __rmul__(self, other): return self * other
    def __rmod__(self, modulus): return self % modulus
    
    def __mod__(self, modulus):
        return LOLA([self.a % modulus, self.b % modulus])
    
    ## SCALING
    
    def scale(self, other, newmod=True): 
        return LOLA([self.a.scale(other, newmod=newmod), self.b.scale(other, newmod=newmod)])
    
    def __rshift__(self, levels=1): # scales down by levels
        return self.scale(self.delta ** levels, newmod=True)
    
    # AUTOMORPHISM
    
    def auto(self, index):
        result = LOLA([self.a.auto(index), self.b.auto(index)])
        return result.keyswitch(index)
    
    def auto_inverse(self):
        result = LOLA([self.a.auto_inverse(), self.b.auto_inverse()])
        return result.keyswitch(conj=True)
    
    def keyswitch(self, index=1, conj=False):
        key = self.ksk[str(index)] if not conj else self.ksk_conj
        result = (key * (self.a % self.Pq_L)).scale(self.q_L)
        return (result % self.modulus) + self.b

    # EN/DECRYPTION
    
    @classmethod
    def encrypt(cls, message, modulus = None):
        modulus = modulus if modulus else cls.q_L # largest modulus by default
        assert type(message) == Poly, "message must be a Poly"
        a = Poly.random(modulus=modulus)
        error = np.sum(np.random.binomial(1, 0.5, size=(cls.N, 2*cls.kappa)), axis=1) 
        e = Poly((error - cls.kappa).astype(int), modulus)
        return LOLA([a, -a * (cls.s % modulus) + e + (message % modulus)])
    
    def decrypt(self, modulus = None):
        q = modulus if modulus else self.modulus
        return (self.b + self.a * (self.s % q)) % q
    
    # MISCELLANEOUS
    
    @property
    def modulus(self): return self.a.modulus
    
    def monomial_shift(self, shift):
        return LOLA([self.a.monomial_shift(shift), self.b.monomial_shift(shift)])
    
    def __repr__(self) -> str:
        return f"{self.a % self.modulus},\n{self.b % self.modulus}"
    
    # BOOTSTRAPPING auxiliaries
    
    def product(self):
        for j in range(1, self.log_h + 1): # product over the buckets
            self = self * self.auto(self.n * self.h // (2**j)) 
            self = self >> 1
        return self # it is faster to scale at the end only
    
    def trace(self):
        auto_index = self.N // 4
        for i in range(log(self.f, 2)):
            self = self + self.auto(auto_index)
            auto_index //= 2
        return self
    
    def angle(self, vec, modulus=0):
        # here it would be best to work with numpy, but we need a certain precision
        vec = np.array(vec, dtype=complex) * (2 * np.pi / self.q)
        vec = (np.cos(vec) + 1j * np.sin(vec)) 
        return self.Encoder.encode_clear(vec, delta=self.boot_delta, modulus=modulus)
    
    def negacyclic_matrix(self):
        # this returns the columns of the negacyclic matrix C (for s*C), still as polynomials
        a = self.a.auto_inverse()
        columns = [a]
        quotient = self.N // self.n
        for j in range(1, self.n):
            a = a << quotient
            for index in range(quotient):
                a[index] = -a[index]
            columns.append(a)
        for j in range(self.n):
            columns[j][0] += self.b[j * quotient]
        return columns
    
    @classmethod
    def block_slice(cls, array, i, matrix=False): # for block keys
        f = cls.f
        if not matrix: # for the secret key
            return [array[i * f + j * cls.blocks + k] for k in range(f) for j in range(cls.h) for _ in range(cls.n)]
        # here the array contains the columns of the negacyclic matrix
        r = cls.Encoder.BR[cls.log_n - 1] # bit-reversed indices of length n
        r = np.concatenate((r, r + (cls.n // 2)))
        if cls.n == 1:
            r = range(1)
        A = [array[col][i * f + j * cls.blocks + k]._integer_() for k in range(f) for j in range(cls.h) for col in r]
        return A

    # BOOTSTRAPPING 
    
    def bootstrap(self):
        C = self.negacyclic_matrix()
        slices = [self.block_slice(C, i, matrix=True) for i in range(2 * self.n)]
        angles = [self.angle(i) for i in slices] 
        products = [i*j for i, j in zip(self.s_block_enc, angles)]
        sum = products[0]
        for i in range(1, 2*self.n):
            sum = sum + products[i]
        sum = sum >> 1
        result = sum.trace().product()
        result = result - result.auto_inverse()
        return self.Encoder.Slot2Coeff(result)
