
import time
import numpy as np
load("sagefhepoly/polyfhe.sage")
load("FHE_DFT_class.sage")

class CKKS:
    @classmethod
    def setup(cls, N, slots, modulus, d=9, r=20, kappa=2, h=2):
        cls.N, cls.log_N = N, log(N, 2)
        assert slots & slots-1 == 0 and N//2 >= slots >= 1, "Wrong number of slots"
        cls.n, cls.log_n = slots, log(slots, 2)
        cls.d, cls.r = d, r
        cls.q = modulus
        cls.prec = log(cls.q, 2) + cls.d + cls.r - 1 
        cls.delta = 2 ** cls.prec
        cls.depth = cls.d + cls.r + 1 + 2 * cls.log_n
        cls.kappa = kappa
        assert h < cls.N, "h must be smaller than N"
        assert h & h-1 == 0, "h must be a power of 2"
        cls.h, cls.log_h = h, ZZ(log(h, 2))
        cls.q_L = cls.q * (2**(cls.prec * cls.depth)) # largest regular modulus (full level)
        cls.Pq_L = cls.q_L * cls.q_L # P = q_L, approximately, as in CKKS
        
        cls.Encoder = FHE_DFT(cls.N, cls.n, cls.q, cls.prec) # see file FHE_DFT_class.sage
        cls.gen_secret(h=cls.h)
        cls.gen_switching_keys()
        cls.boot_precompute()
        
    @staticmethod
    def gaussian_integer(value):
        return round(value.real()) + I * round(value.imag())
    
    @classmethod
    def gaussian_poly(cls, value):
        c0 = Poly(set_ntl([round(value.real())], cls.q_L), cls.q_L)
        c1 = Poly(set_ntl([round(value.imag())], cls.q_L), cls.q_L)
        return c0 + c1.monomial_shift(cls.N // 2)
        
    @classmethod
    def gen_secret(cls, h):
        cls.s = np.concatenate((np.ones(h), np.zeros(cls.N - h - 1)))
        np.random.shuffle(cls.s)
        cls.s = np.append([1], cls.s)
        cls.s = Poly(cls.s, 0)
        
    @classmethod
    def gen_switching_keys(cls):
        assert hasattr(cls, "s"), "secret key not generated"
        # the evaluation key for ciphertext multiplication, encrypting s^2
        evk = Poly(((cls.s % cls.Pq_L) ** 2) * cls.q_L, cls.q_L)
        cls.evk = CKKS.encrypt(evk, cls.Pq_L)
        
        # we generate the keyswitching keys for the automorphisms of powers of 2, with pos. and neg. sign
        cls.ksk = {}
        for index in [1 << i for i in range(cls.N.bit_length()-2)]:
            for j in range(-1, 2, 2): # j = -1, 1
                newkey = Poly((cls.s.auto(index * j) * cls.q_L) % cls.Pq_L, cls.Pq_L)
                cls.ksk[str(index * j)] = CKKS.encrypt(newkey, cls.Pq_L)
        # ...and for the conjugation automorphism        
        newkey =  Poly((cls.s.auto_inverse() * cls.q_L) % cls.Pq_L, cls.Pq_L)
        cls.ksk_conj = CKKS.encrypt(newkey, cls.Pq_L)
        
    @classmethod
    def boot_precompute(cls):
        cls.scaling_poly = Poly(set_ntl([round(cls.delta * cls.n / cls.N)], cls.q_L), cls.q_L)
        cls.fac = (cls.q / (2*pi*cls.delta * 2 / cls.n)) ** (1 / (2**cls.r))

        cls.coeffs_taylor = [cls.fac*cls.delta*(2*pi*I*cls.delta/ (2**cls.r) / cls.q) ** j /factorial(j) for j in range(cls.d + 1)]
        cls.coeffs_taylor = [cls.gaussian_poly(cls.gaussian_integer(c)) for c in cls.coeffs_taylor]
        for i in reversed(range(cls.d + 1)): # we set the right moduli
            cls.coeffs_taylor[i] %= cls.q_L // (cls.delta ** (cls.d - i + 1 + cls.log_n))
        
    ## INIT    
        
    def __init__(self, args) -> None:
        self.a, self.b = args
    
    # ARITHMETIC OPERATIONS

    def __add__(self, other):
        if isinstance(other, Poly): return CKKS([self.a, self.b + other])
        return CKKS([self.a + other.a, self.b + other.b])
     
    def __neg__(self): return CKKS([-self.a, -self.b])

    def __sub__(self, other):
        if isinstance(other, Poly): return CKKS([self.a, self.b - other])
        return CKKS([self.a - other.a, self.b - other.b])


    def __mul__(self, other):
        if isinstance(other, CKKS):
            assert self.a.modulus == other.a.modulus, f"Moduli must be the same (and = {self.q})"    
            d0 = (self.a * other.a) % self.Pq_L
            d1 = self.a * other.b + self.b * other.a
            d2 = self.b * other.b
            d0 = (self.evk * d0).scale(self.q_L) % d1.modulus
            return (CKKS([d1, d2]) + d0)
        assert not isinstance(other, int), "int multiplication not implemented"
        
        m1, m2 = self.modulus, other.modulus # Below, we figure the right modulus to continue
        if m1 == 0 and m2 != 0: self = self % m2
        elif m2 == 0 and m1 != 0: other = other % m1
        elif m2 != 0 and m1 != 0:
            modulus = min(m1, m2)
            self, other = self % modulus, other % modulus            
        return CKKS([self.a * other, self.b * other]) 
    
    def __radd__(self, other): return self + other
    def __rsub__(self, other): return self - other
    def __rmul__(self, other): return self * other
    def __rmod__(self, modulus): return self % modulus
    
    def __mod__(self, modulus):
        return CKKS([self.a % modulus, self.b % modulus])
    
    ## SCALING
    
    def scale(self, other, newmod=True): 
        return CKKS([self.a.scale(other, newmod=newmod), self.b.scale(other, newmod=newmod)])
    
    def __rshift__(self, levels=1): # scales down by levels
        return self.scale(self.delta ** levels, newmod=True)
    
    # AUTOMORPHISM
    
    def auto(self, index):
        result = CKKS([self.a.auto(index), self.b.auto(index)])
        return result.keyswitch(index)
    
    def auto_inverse(self):
        result = CKKS([self.a.auto_inverse(), self.b.auto_inverse()])
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
        return CKKS([a, -a * (cls.s % modulus) + e + (message % modulus)])
    
    def decrypt(self, modulus = None):
        q = modulus if modulus else self.modulus
        return (self.b + self.a * (self.s % q)) % q
    
    # MISCELLANEOUS
    
    @property
    def modulus(self): return self.a.modulus
    
    def monomial_shift(self, shift):
        return CKKS([self.a.monomial_shift(shift), self.b.monomial_shift(shift)])
    
    def __repr__(self) -> str:
        return f"{self.a % self.modulus},\n{self.b % self.modulus}"
    
    # BOOTSTRAPPING auxiliaries
    
    def trace(self):
        auto_index = self.N // 4
        param = 1 if self.n == 1 else self.log_n
        for i in range(self.log_N - param):
            self = self + self.auto(auto_index)
            auto_index //= 2
        return self
    
    def horner(self):
        u = self.coeffs_taylor[-1]
        for i in range(self.d):
            u = self * u 
            u = u >> 1
            u = u + self.coeffs_taylor[-2 - i]
            self %= self.modulus // self.delta
        return u
    
    def evaluate(self):
        self = self.horner()
        for i in range(self.r):
            self = self * self
            self = self >> 1
        return self - self.auto_inverse()
        
    # BOOTSTRAPPING 
    
    def bootstrap(self):
        assert self.modulus == self.q
        self %= self.q_L # ModRaise
        self = self.trace()
        if self.n == 1:
            self = self + self.auto_inverse()
        self = (self * self.scaling_poly) >> 1
        if self.n > 1:
            self = self.Encoder.Coeff2Slot(self)
        result = self.evaluate()
        return self.Encoder.Slot2Coeff(result)
