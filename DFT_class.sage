
import numpy as np
from mpmath import mp, mpc
load("sagefhepoly/polyfhe.sage")
load("root_of_unity.sage")

def br(bit, length): # bit-reversal
    return int('{:0{width}b}'.format(bit, width=length)[::-1], 2)

class DFT: # a module to perform the DFT in the clear
    roots = None
    
    def __init__(self, N, precision=53) -> None:
        self.N = N
        self.prec = precision
        if precision > 53:
            mp.dps = precision * log(2, 10)
        assert N & (N - 1) == 0, "N must be a power of 2"
        self.log_N = ZZ(log(N, 2))

        self.gen_BR_lookup()
        self.gen_roots()
        self.encoding_precomputation()
        
    def gen_BR_lookup(self):
        # generates a look-up table for the bit-reversal permutation
        assert hasattr(self, 'N'), "N is not defined, initialize the DFT class first"
        self.BR = [np.array([br(i, l) for i in range(2**l)], dtype=int) for l in range(self.log_N + 1)]
                
    def gen_roots(self): # 2 MB for N = 2^15
        self.roots_BR     = [[]] + [[0] * 2**(k-1) for k in range(1, self.log_N)]
        self.roots        = [[]] + [[0] * 2**(k-1) for k in range(1, self.log_N)]
        self.roots_BR_inv = [[]] + [[0] * 2**(k-1) for k in range(1, self.log_N)]
        for k in range(1, self.log_N):
            step = self.N // 2**(k+1)
            self.roots_BR[k]     = [root(2*self.N, 5**(self.BR[k-1][i]) * step, self.prec) for i in range(2**(k-1))]
            self.roots_BR_inv[k] = [root(2*self.N,-5**(self.BR[k-1][i]) * step, self.prec) for i in range(2**(k-1))]
            self.roots[k]        = [root(2*self.N, 5**(i)               * step, 53) for i in range(2**(k-1))]
        DFT.roots = self.roots
            
    def convert_poly_to_C(self, poly, precision: int=53):
        # converting a polynomial to C representation with the appropriate precision
        if precision > 53:
            return np.array([mpc(ZZ(i)) for i in poly.list(full = True)], dtype=object)
        else:
            return np.array([ZZ(i) for i in poly.list(full = True)], dtype=complex)
            
    def decode_no_to_bo(self, poly, precision: int=53): # 369ms
        # decoding, normal to bit-reversed order
        p = self.convert_poly_to_C(poly, precision)
        v = p[:self.N//2] + p[self.N//2:] * 1j
        for k in range(1, self.log_N):
            step = self.N // 2**(k+1)
            for i in range(2**(k-1)):
                for j in range(step):
                    index = i * step * 2 + j
                    a = v[index]
                    prod = self.roots_BR[k][i] * v[index + step]
                    v[index] = a + prod
                    v[index + step] = a - prod        
        return v
    
    def decode_bo_to_no(self, poly, precision: int=53): # 353ms
        # decoding, bit-reversed to normal order
        p = self.convert_poly_to_C(poly, precision)
        v = p[::2] + p[1::2] * 1j
        for k in range(1, self.log_N):
            step = self.N // 2**(k+1)
            shift1, shift2 = 2**(k-1), 2**k
            for i in range(shift1):
                for j in range(step):
                    index = i + j * shift2
                    a = v[index]
                    prod = self.roots[k][i] * v[index + shift1]
                    v[index] = a + prod
                    v[index + shift1] = a - prod
        return v
    
    def encode_bo_to_no(self, vect, precision: int=53): # 817ms
        # encoding, bit-reversed to normal order, OVERWRITES the input vector
        v = vect
        assert len(v) == self.N // 2, "Input vector has the wrong length"
        assert type(v) == np.ndarray, "Input vector must be a numpy array"
        for k in reversed(range(1, self.log_N)):
            for i in range(2**(k-1)):
                step = self.N // 2**(k+1)
                for j in range(step):
                    index = i * step * 2 + j
                    a = v[index]
                    b = v[index + step]
                    v[index] = (a + b)
                    v[index + step] = (a - b) * self.roots_BR_inv[k][i]
            v /= 2
        return np.append(v.real, v.imag)
    
    def encode_no_to_no(self, vect, precision: int=53): # 827ms
        # encodes a normal order vector to normal order, by applying bit-reversal and then encoding
        v = vect
        assert type(v) == np.ndarray, "Input vector must be a numpy array"
        lg = ZZ(log(len(v), 2))
        reverse = np.array([v[br(i, lg)] for i in range(len(v))], dtype=type(v[0]))
        return self.encode_bo_to_no(reverse, precision)
    
    def encode_fast(self, vec, bitrev: bool=False): # 1 ms
        assert type(vec) == np.ndarray, "Input vector must be a numpy array"
        assert hasattr(self, 'sequence'), "Precomputation not done, call pre() first"
        if not bitrev: vec = vec[self.BR[-2]]
        #if not bitrev: vec = vec[self.BR[log(len(vec), 2)]]
        delta, index = 2, 1
        for _ in range(self.log_N - 1):
            vec = np.reshape(vec, (self.N//delta, delta//2))
            a, b = vec[::2], vec[1::2]
            vec[::2], vec[1::2] = (a + b), (a - b) * self.sequence[index]
            vec /= 2
            delta *= 2
            index += 1
        vec = np.reshape(vec, self.N//2)
        return np.append(vec.real, vec.imag)
    
    def encoding_precomputation(self):
        delta, list_roots = 2, []
        for k in range(self.log_N - 1, 0, -1):
            for i in range(self.N // 2 // delta):
                u = root(2*self.N, (5 ** self.BR[k-1][i]) * self.N // (2**(k+1)))
                list_roots.append(u)
            delta *= 2
        sequence, index, delta = [[]], 0, 2
        for _ in range(self.log_N - 1):
            step = self.N // 2 // delta
            A = [[list_roots[i] ** (-1)] for i in range(index, index + step)]
            sequence.append(np.array(A))
            index += step
            delta *= 2
        self.sequence = sequence
