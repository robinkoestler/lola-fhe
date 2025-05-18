
load("DFT_class.sage")

def next_power_of_2(x):
    return 1 << (floor(x) - 1).bit_length()

class FHE_DFT(DFT):
    def __init__(self, N, slots, modulus, precision=53) -> None:
        super().__init__(N, precision)
        assert slots & (slots - 1) == 0, "slots must be a power of 2"
        assert slots <= N//2, "slots must be <= N/2"
        assert hasattr(Poly, "N"),  "Poly module not loaded"
        self.delta = 2 ** precision
        self.n = slots
        self.log_n = ZZ(log(slots, 2))
        self.mod = modulus
        if self.n > 1:
            self.precompute()
    
    # this will be applied to encode plaintext values into the polynomial ring
    def encode_clear(self, values, delta=None, modulus = None, bitrev: bool=False): # 1.2 ms
        if type(values) == list:
            values = np.array(values, dtype=complex)
        assert type(values) == np.ndarray, "Input vector must be a numpy array"
        modulus = self.mod if modulus is None else modulus
        complex_encoding = super().encode_fast(values, bitrev=bitrev)
        maximum = np.max(np.abs(complex_encoding))
        d = self.delta if delta is None else delta
        if maximum * d >= 2**63 - 1:
            # Encoding is too large for numpy's int64
            # to maintain speed, we first scale up the CC(64) values by the maximum
            first_scale = (2**62) // next_power_of_2(maximum)
            complex_encoding = (complex_encoding * first_scale).astype(int)
            p = Poly(set_ntl(complex_encoding, modulus), modulus)
            return p * (d // first_scale) # and now we scale the rest in NTL
        int_array = np.round(complex_encoding * d).astype(int)
        return Poly(set_ntl(int_array, modulus), modulus)
    
    def precompute(self, scaling = True):
        # We create the polynomial encoding for the primitive roots
        # First, for the Slot2Coeff precomputation
        v = [[0, 0, 0] for _ in range(self.log_n)]
        scale = 1/(2*I) if scaling else 1
        li1 = [scale     for _ in range(self.n // 2)]
        li2 = [scale * I for _ in range(self.n // 2)]
        v[0][0] = (li1 + li2) * (self.N // self.n // 2)
        v[0][1] = (li2 + li1) * (self.N // self.n // 2)
        v[0] = [self.encode_clear(i, modulus=0) for i in v[0][:2]] + [0]
        
        delta, s = self.N // 2, 1 # s = "scale"
        for k in range(1, self.log_n):
            if k == self.log_n - 1 and scaling: s = 2 / self.n 
            r = super().roots[k]
            v[k][0] = [[s, -r[i] * s][j] for _ in range(delta // 2) for j in range(2) for i in range(self.N//2//delta)]
            v[k][1] = [[r[i] * s, 0] [j] for _ in range(delta // 2) for j in range(2) for i in range(self.N//2//delta)]
            v[k][2] = [[0, s]        [j] for _ in range(delta // 2) for j in range(2) for i in range(self.N//2//delta)]
            v[k] = [self.encode_clear(i, modulus=0) for i in v[k]]
            delta //= 2
        self.poly_roots_S2C = v
        
        # Now for the Coeff2Slot precomputation
        v = [[0, 0, 0] for _ in range(self.log_n)]
        li1 = ([0.5] * (self.n//2) + [0.5*I] * (self.n//2)) * (self.N // self.n // 2)
        li2 = ([0.5] * (self.n//2) + [-0.5*I] * (self.n//2)) * (self.N // self.n // 2)
        v[0] = [self.encode_clear(i, modulus=0) for i in [li1, li2]] + [0]
        for k in range(self.log_n-1, 0, -1):
            delta = self.N // (2**k)
            r = super().roots[k]
            v[k][0] = [[0.5, -0.5 / r[i]][j] for _ in range(delta // 2) for j in range(2) for i in range(self.N//2//delta)]
            v[k][1] = [[0.5, 0]          [j] for _ in range(delta // 2) for j in range(2) for i in range(self.N//2//delta)]
            v[k][2] = [[0  , 0.5 / r[i]] [j] for _ in range(delta // 2) for j in range(2) for i in range(self.N//2//delta)]
            v[k] = [self.encode_clear(i, modulus=0) for i in v[k]]
        self.poly_roots_C2S = v
    
    
    ## this is the ONLY method, that will be used for bootstrapping
    def Slot2Coeff(self, poly):
        # homomorphic Slot2Coeff with input in bit-reversed order
        p = poly
        if self.n == 1: # dividing by I
            return -p.monomial_shift(self.N//2)
        assert type(p) == Poly or type(p.a) == Poly, "Input must be a polynomial/RLWE/CKKS"
        p = p * self.poly_roots_S2C[0][0] + p.auto(self.n // 2) * self.poly_roots_S2C[0][1]
        p = p >> 1
        for k in range(1, self.log_n):
            v0, v1, v2 = self.poly_roots_S2C[k][0], self.poly_roots_S2C[k][1], self.poly_roots_S2C[k][2]
            step = 2 ** (k-1)
            p = p * v0 + p.auto(step) * v1 + p.auto(-step) * v2
            p = p >> 1
        return p
    
    # may not work properly
    def Coeff2Slot(self, poly): # 168 ms
        # homomorphic Coeff2Slot with output in ?? order
        p = poly
        if self.n == 1: return poly.monomial_shift(self.N//2)
        assert type(p) == Poly or type(p.a) == Poly, "Input must be a polynomial/RLWE/CKKS"
        for k in range(self.log_n-1, 0, -1):
            v0, v1, v2 = self.poly_roots_C2S[k][0], self.poly_roots_C2S[k][1], self.poly_roots_C2S[k][2]
            step = 2 ** (k-1)
            p = (p * v0 + p.auto(step) * v1 + p.auto(-step) * v2)
            p = p >> 1
        p = p.auto_inverse() * self.poly_roots_C2S[0][0] + p * self.poly_roots_C2S[0][1]
        return p >> 1
